"""
GSM8K 推理策略对比实验（E1-E5）

策略:
  E1 Zero-shot 基线          (temp=0.0, 1次)
  E2 Zero-shot CoT           (temp=0.0, 1次)
  E3 Few-shot CoT            (temp=0.0, 1次)
  E4 Zero-shot CoT + 自洽性  (temp=0.7, 5次投票)
  E5 Few-shot CoT + 自洽性   (temp=0.7, 5次投票)

用法:
  python experiment.py --limit 30                 # 本地模型跑30题
  python experiment.py --limit 2 --max-tokens 128 # 快速冒烟测试
  # 启用 API 模型（需先设置密钥）:
  #   set DEEPSEEK_API_KEY=sk-xxx 并可选 DEEPSEEK_BASE_URL
  python experiment.py --models qwen2.5-3b,deepseek-v4-flash
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from api_client import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, get_api_generator

# 必须在 import transformers 之前设置离线环境（模型已缓存到本地）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "gsm8k_test_100.jsonl"
RESULT_FILE = BASE_DIR / "results.json"

# ---------------- 1. 数据 ----------------

def load_questions(limit=30):
    items = []
    with DATA_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit]


def gold_answer(item):
    m = re.search(r"####\s*([\d.]+)", item["answer"])
    return float(m.group(1)) if m else None


# ---------------- 2. 提示词与策略 ----------------

FEW_SHOT_EXAMPLES = """
问题：Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh egg. How much in dollars does she make every day at the farmers' market?
解答：每天产16个蛋。早餐吃3个，烘焙用4个，共用掉7个。剩余16-7=9个。每个卖2美元，收入9*2=18美元。#### 18

问题：A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
解答：蓝纤维2卷。白纤维是蓝纤维的一半，即2/2=1卷。总共2+1=3卷。#### 3
"""


def build_prompts(question):
    """E1/E2 共用问题本体，E3 在问题前拼接2个固定例题。"""
    return {
        "E1": f"问题：{question}\n请直接给出最终答案，并用 #### 数字 结尾。",
        "E2": f"问题：{question}\n请一步步思考，最后用 #### 数字 给出答案。",
        "E3": FEW_SHOT_EXAMPLES + "\n问题：" + question + "\n请按照上面格式解答",
    }


# (策略名 -> 用的提示词, 采样次数, 温度)
STRATEGIES = {
    "E1": {"prompt": "E1", "n": 1, "temperature": 0.0},
    "E2": {"prompt": "E2", "n": 1, "temperature": 0.0},
    "E3": {"prompt": "E3", "n": 1, "temperature": 0.0},
    "E4": {"prompt": "E2", "n": 5, "temperature": 0.7},
    "E5": {"prompt": "E3", "n": 5, "temperature": 0.7},
}
STRATEGY_LABELS = {
    "E1": "Zero-shot",
    "E2": "Zero-shot CoT",
    "E3": "Few-shot CoT",
    "E4": "CoT+SC",
    "E5": "Few-shot+SC",
}

# ---------------- 3. 答案提取与自洽性投票（复用用户给定逻辑） ----------------

def extract_answer(text):
    """从模型输出中提取 #### 后面的数字"""
    match = re.search(r'####\s*([\d.]+)', text)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def self_consistency_vote(model_func, prompt, n=5):
    """对同一个prompt采样n次，返回出现次数最多的答案"""
    answers = []
    for _ in range(n):
        response = model_func(prompt, temperature=0.7)  # 提高温度增加多样性
        ans = extract_answer(response)
        if ans is not None:
            answers.append(ans)
    if not answers:
        return None, 0  # 无有效答案
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    return most_common[0], most_common[1]  # (答案, 投票数)


# ---------------- 4. 模型接入 ----------------

_local = {}  # 懒加载本地模型的缓存


def load_local_model():
    if _local:
        return _local
    print("Loading local model unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit ...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    model_name = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "quantization_config"):
        config.quantization_config["llm_int8_enable_fp32_cpu_offload"] = True

    max_memory = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {gpu_mem:.1f} GB")
        max_memory = {0: f"{max(1, int(gpu_mem * 0.85))}GiB", "cpu": "64GiB"}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    print("Local model loaded successfully!")
    _local["tokenizer"] = tokenizer
    _local["model"] = model
    return _local


def generate_local(prompt, temperature=0.0, max_new_tokens=512):
    import torch
    tokenizer = _local["tokenizer"]
    model = _local["model"]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0  # 温度<=0 时走贪心解码（transformers 要求 do_sample=False）
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
    )


# ---------------- 5. 主流程 ----------------

def run_question(model_key, gen_func, question, prompts, max_new_tokens):
    """对单个问题运行全部 E1-E5，返回 (results_dict, total_generations)。"""
    out = {}
    for name, cfg in STRATEGIES.items():
        prompt = prompts[cfg["prompt"]]
        if cfg["n"] == 1:
            response = gen_func(prompt, temperature=cfg["temperature"], max_new_tokens=max_new_tokens)
            ans = extract_answer(response)
            out[name] = {"answer": ans, "samples": [response]}
        else:
            # 包装器：记录每次采样文本，同时保证只生成 n 次（复用 self_consistency_vote）
            samples_log = []

            def tracked(p, temperature=cfg["temperature"]):
                resp = gen_func(p, temperature=temperature, max_new_tokens=max_new_tokens)
                samples_log.append(resp)
                return resp

            ans, votes = self_consistency_vote(tracked, prompt, n=cfg["n"])
            out[name] = {
                "answer": ans,
                "votes": votes,
                "sample_answers": [extract_answer(s) for s in samples_log],
                "samples": samples_log,
            }
    return out


def build_report(results, questions, models_meta, limit):
    """计算各模型/策略准确率并打印最终产出表。"""
    print("\n" + "=" * 78)
    print(f"FINAL RESULTS (accuracy = correct / {limit})")
    print("=" * 78)
    header = f"{'Model':<20}" + "".join(f"{STRATEGY_LABELS[k]:>14}" for k in STRATEGIES)
    print(header)
    print("-" * 78)

    report = {}
    for key in models_meta:
        if key not in results["models"]:
            continue
        model_res = results["models"][key]
        report[key] = {}
        for name in STRATEGIES:
            correct = 0
            for q_res in model_res.values():
                if q_res.get(name, {}).get("answer") is not None and \
                   abs(q_res[name]["answer"] - q_res["gold"]) < 1e-9:
                    correct += 1
            report[key][name] = correct / limit
        accs = "".join(f"{report[key][k]*100:13.1f}%" for k in STRATEGIES)
        print(f"{models_meta[key]['label']:<20}{accs}")

    # 附加记录：E4/E5 每题的投票有效答案数统计
    print("-" * 78)
    for key in models_meta:
        if key not in results["models"]:
            continue
        model_res = results["models"][key]
        for name in ("E4", "E5"):
            n_valid = []
            for q_res in model_res.values():
                votes = q_res.get(name, {}).get("votes", 0)
                n_valid.append(votes)
            avg = sum(n_valid) / len(n_valid) if n_valid else 0.0
            min_v = min(n_valid) if n_valid else 0
            print(f"[note] {models_meta[key]['label']} {STRATEGY_LABELS[name]} avg valid votes: "
                  f"{avg:.2f} / 5 (min {min_v}, shows output parsability)")
    return report


def main():
    parser = argparse.ArgumentParser(description="GSM8K E1-E5 推理策略实验")
    parser.add_argument("--models", default="qwen2.5-3b",
                        help="逗号分隔: qwen2.5-3b,deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=30, help="跑多少题（默认30）")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="忽略已有结果文件重新跑")
    parser.add_argument("--output", default=None,
                        help="结果文件路径（默认 results.json；多模型并行评测时请指定不同文件）")
    args = parser.parse_args()

    questions = load_questions(args.limit)
    print(f"Loaded {len(questions)} GSM8K test questions (from {DATA_FILE.name})")

    # 模型注册表
    MODELS_META = {
        "qwen2.5-3b": {"kind": "local", "label": "Local Qwen2.5-3B"},
        "deepseek-v4-flash": {"kind": "deepseek", "label": "DeepSeek-V4-Flash",
                              "model": DEEPSEEK_MODEL,
                              "base_url": os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)},
    }

    # 解析要跑的模型并检查密钥
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    to_run = []
    for m in requested:
        meta = MODELS_META[m]
        if meta["kind"] == "deepseek":
            if not os.environ.get("DEEPSEEK_API_KEY"):
                print(f"[skip] {meta['label']}: env var DEEPSEEK_API_KEY not set")
                continue
        to_run.append((m, meta))
    if not to_run:
        print("No runnable models, exit.")
        return

    # 结果文件（断点续跑；--output 可指定独立文件，避免并行评测时互相覆盖）
    result_file = Path(args.output) if args.output else RESULT_FILE
    results = {"models": {}}
    if result_file.exists() and not args.force:
        try:
            with result_file.open(encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {"models": {}}

    for key, meta in to_run:
        if meta["kind"] == "local":
            load_local_model()
            gen_func = lambda p, temperature=0.0, max_new_tokens=512: generate_local(
                p, temperature=temperature, max_new_tokens=max_new_tokens)
        elif meta["kind"] == "deepseek":
            api_key = os.environ["DEEPSEEK_API_KEY"]
            gen_func = get_api_generator(meta["base_url"], api_key, meta["model"])
        else:
            continue

        model_res = results["models"].setdefault(key, {})
        print(f"\nEvaluating model: {meta['label']} ({len(questions)} questions)")
        t_start = time.time()
        for i, item in enumerate(questions):
            qi = str(i)
            done = qi in model_res and all(s in model_res[qi] for s in STRATEGIES)
            if done and not args.force:
                continue
            prompts = build_prompts(item["question"])
            gold = gold_answer(item)
            q_res = run_question(key, gen_func, item["question"], prompts, args.max_tokens)
            q_res["gold"] = gold
            model_res[qi] = q_res
            # 每题结束立即落盘
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            elapsed = time.time() - t_start
            ans = q_res["E5"]["answer"]
            status = "OK" if ans is not None else "--"
            print(f"  [{key}] q{i+1}/{len(questions)} E5 answer={status} "
                  f"(elapsed {elapsed/60:.1f} min)")
        print(f"{meta['label']} done in {(time.time()-t_start)/60:.1f} min")

    build_report(results, questions, MODELS_META, args.limit)


if __name__ == "__main__":
    main()

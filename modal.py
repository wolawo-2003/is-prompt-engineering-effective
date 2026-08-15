import os
import sys

# 强制 stdout 使用 UTF-8，避免管道重定向时中文乱码（Code Runner 输出窗格按 UTF-8 解码）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 强制离线模式：必须在 import transformers 之前设置，
# 因为 huggingface_hub 在 import 时会缓存离线标志（模型已缓存到本地）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import torch

model_name = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit"

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("正在加载配置...")
config = AutoConfig.from_pretrained(model_name)
if hasattr(config, "quantization_config"):
    config.quantization_config["llm_int8_enable_fp32_cpu_offload"] = True
    print("已启用CPU offload支持")

max_memory = None
try:
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"检测到GPU显存: {gpu_mem:.1f} GB")
        max_memory = {0: f"{max(1, int(gpu_mem * 0.85))}GiB", "cpu": "64GiB"}
    else:
        print("未检测到GPU，将使用CPU运行（速度较慢）")
except Exception as e:
    print(f"检测GPU信息失败: {e}")

import tempfile
offload_folder = tempfile.mkdtemp(prefix="model_offload_")
print(f"磁盘offload目录: {offload_folder}")

print("正在加载4-bit模型，请稍候...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    max_memory=max_memory,
    offload_folder=offload_folder,
    low_cpu_mem_usage=True
)
print("模型加载成功！")

messages = [
    {"role": "user", "content": "你好，请介绍一下你自己。"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("\n模型正在生成回复...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1
)

response = tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"\n模型回复：\n{response}")

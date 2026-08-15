"""
DeepSeek API 客户端（OpenAI 兼容接口），纯标准库实现，无第三方依赖。

用法:
    from api_client import api_chat_completion, get_api_generator

    # 1. 直接调用
    text = api_chat_completion("https://api.deepseek.com/v1", "sk-xxx",
                               "deepseek-v4-flash", "hello", temperature=0.0, max_new_tokens=512)

    # 2. 包装成与本地模型一致的生成函数签名: (prompt, temperature, max_new_tokens) -> text
    gen = get_api_generator("https://api.deepseek.com/v1", "sk-xxx", "deepseek-v4-flash")
    text = gen("hello", temperature=0.7, max_new_tokens=512)

    # 3. 直接读环境变量构造（配合 experiment.py 的 --models 参数使用）
    gen = get_deepseek_generator()   # 需设置 DEEPSEEK_API_KEY
"""

import json
import os
import urllib.request

# DeepSeek 官方接口（当前使用模型 deepseek-v4-flash）
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"


def api_chat_completion(base_url, api_key, model_name, prompt, temperature, max_new_tokens):
    """调用 OpenAI 兼容的 /chat/completions 接口，返回回复文本。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def get_api_generator(base_url, api_key, model_name):
    """包装成与本地模型 generate_local 相同的调用签名，便于直接替换。"""
    def generate(prompt, temperature=0.0, max_new_tokens=512):
        return api_chat_completion(base_url, api_key, model_name, prompt, temperature, max_new_tokens)
    return generate


def get_deepseek_generator():
    """从环境变量 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL 构造 DeepSeek 生成函数。"""
    api_key = os.environ["DEEPSEEK_API_KEY"]
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
    return get_api_generator(base_url, api_key, DEEPSEEK_MODEL)

"""
deployment/vllm/openai-api.py
vLLM OpenAI 兼容 API 调用示例

支持两种模式：
1. vLLM 服务器模式（先启动 start-vllm.sh）
2. 直接本地模型模式（不需要 vLLM）
"""

import os

# ============================================================
# 模式一：调用 vLLM 部署的 API（需要先运行 start-vllm.sh）
# ============================================================
def call_vllm_api():
    from openai import OpenAI

    client = OpenAI(
        api_key="mindhealer-secret-key-2024",
        base_url="http://localhost:8000/v1",
    )

    response = client.chat.completions.create(
        model="mindhealer-gemma4-psych",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位专业、温暖、有同理心的心理咨询师。"
                )
            },
            {
                "role": "user",
                "content": "最近总是失眠，是怎么回事？"
            }
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    print(response.choices[0].message.content)


# ============================================================
# 模式二：直接用 transformers 加载本地合并模型
# ============================================================
def call_local_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_path = "./outputs/merged/gemma4-psych-consultant"

    print(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    prompt = """<bos><start_of_turn>system
你是一位专业、温暖、有同理心的心理咨询师。<end_of_turn>
<start_of_turn>user
最近总是失眠，是怎么回事？<end_of_turn>
<start_of_turn>model
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python openai-api.py [vllm|local]")
        print("  vllm  - 调用 vLLM API（需先启动 start-vllm.sh）")
        print("  local - 直接加载本地模型")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "vllm":
        call_vllm_api()
    elif mode == "local":
        call_local_model()
    else:
        print(f"未知模式: {mode}")
        sys.exit(1)

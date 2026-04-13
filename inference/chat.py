"""
inference/chat.py
命令行对话脚本，测试微调效果

用法:
  python inference/chat.py

需要在有 vLLM 服务运行的情况下使用，
或者修改 MODEL_PATH 为合并后的本地模型路径。
"""

import os
from openai import OpenAI

# 默认配置
API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-4-E4B-it")

# 如果想直接用本地模型（不用 vLLM），改成这个：
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)


def chat(user_input, system_prompt=None, history=None):
    """
    发送对话请求
    """
    messages = []

    # system prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif not system_prompt:
        # 默认心理咨询师设定
        messages.append({
            "role": "system",
            "content": (
                "你是一位专业、温暖、有同理心的心理咨询师。你擅长理性情绪行为疗法（REBT），"
                "能够通过提问引导来访者自我探索，帮助他们识别非理性信念，找到建设性的应对方式。"
                "你的回复应该温和、专业、富有共情，同时避免给出直接的评判或建议。"
            )
        })

    # 对话历史
    if history:
        for h in history:
            messages.append(h)

    # 当前输入
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=False,
    )

    return response.choices[0].message.content


def chat_loop():
    """
    交互式对话循环
    """
    print("=" * 50)
    print("MindHealer 心理咨询对话（输入 quit 退出）")
    print("=" * 50)
    print()

    history = []

    while True:
        try:
            user_input = input("来访者 > ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("感谢来访，下次见！")
            break

        # 添加历史
        history.append({"role": "user", "content": user_input})

        print("咨询师 > ", end="", flush=True)
        try:
            reply = chat(user_input, history=history)
            print(reply)
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"\n[错误] {e}")
            print("请确认 vLLM 服务是否启动（bash deployment/vllm/start-vllm.sh）")
            break

        print()


if __name__ == "__main__":
    print(f"API Base: {API_BASE}")
    print(f"Model: {MODEL_PATH}")
    print()
    chat_loop()

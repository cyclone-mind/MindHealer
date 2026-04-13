"""
inference/web_demo.py
Gradio 网页界面，快速演示用

用法:
  python inference/web_demo.py
"""

import gradio as gr
from openai import OpenAI
import os

API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-4-E4B-it")

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

SYSTEM_PROMPT = (
    "你是一位专业、温暖、有同理心的心理咨询师。你擅长理性情绪行为疗法（REBT），"
    "能够通过提问引导来访者自我探索，帮助他们识别非理性信念，找到建设性的应对方式。"
    "你的回复应该温和、专业、富有共情，同时避免给出直接的评判或建议。"
)


def predict(message, history):
    """
    Gradio 的预测函数
    history 是 [(user, assistant), ...] 的列表
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True,
        )

        # 流式输出
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                yield full_response

    except Exception as e:
        yield f"[错误] {e}\n\n请确认 vLLM 服务是否启动：bash deployment/vllm/start-vllm.sh"


# Gradio 界面
demo = gr.ChatInterface(
    fn=predict,
    title="MindHealer — 心理咨询大模型",
    description=(
        "基于 Google Gemma-4 微调的心理咨询对话模型。"
        "如果你感到不适或处于危机状态，请寻求专业帮助。"
    ),
    theme="soft",
    retry_btn="重试",
    undo_btn="删除上一条",
    clear_btn="清空对话",
    examples=[
        ["最近工作压力很大，总是很焦虑"],
        ["我和男朋友吵架了，不知道该怎么办"],
        ["觉得自己什么都做不好，很迷茫"],
    ],
)

if __name__ == "__main__":
    print(f"启动 Gradio 界面...")
    print(f"API Base: {API_BASE}")
    print(f"Model: {MODEL_PATH}")
    print(f"访问 http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)

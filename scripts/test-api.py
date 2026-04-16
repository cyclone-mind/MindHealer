#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MindHealer API 测试脚本
测试部署的 vLLM/Ollama API 服务是否正常工作

用法:
  python scripts/test-api.py                        # 使用默认配置测试 vLLM
  python scripts/test-api.py --type ollama          # 测试 Ollama
  python scripts/test-api.py -u http://localhost:8000 -k my-key -m my-model
"""

import sys
import argparse
from typing import Optional
from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = "你是一位专业、温暖、有同理心的心理咨询师。"

DEFAULT_TEST_CASES = [
    {
        "role": "user",
        "content": "我最近总是失眠，是怎么回事？"
    },
    {
        "role": "user", 
        "content": "工作压力很大，总是很焦虑怎么办？"
    },
    {
        "role": "user",
        "content": "和女朋友吵架了，心情很低落"
    },
]


def test_chatCompletion(client: OpenAI, model_name: str, messages: list, 
                        temperature: float = 0.7, max_tokens: int = 512) -> bool:
    """测试 chat completions API"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content
        print(f"  回复: {content[:150]}...")
        print(f"  Tokens: prompt={response.usage.prompt_tokens}, "
              f"completion={response.usage.completion_tokens}, "
              f"total={response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"  ❌ 请求失败: {e}")
        return False


def test_streaming(client: OpenAI, model_name: str, messages: list) -> bool:
    """测试流式输出"""
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=128,
        )
        
        print("  流式输出: ", end="", flush=True)
        full_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_content += text
        print()
        print(f"  ✅ 流式输出正常 ({len(full_content)} 字符)")
        return True
        
    except Exception as e:
        print(f"  ❌ 流式输出失败: {e}")
        return False


def run_tests(base_url: str, api_key: Optional[str], model_name: str, 
              use_streaming: bool = True):
    """运行所有测试"""
    
    print("=" * 60)
    print("MindHealer API 测试")
    print("=" * 60)
    print(f"API 地址: {base_url}")
    print(f"模型名: {model_name}")
    print(f"流式输出: {'开启' if use_streaming else '关闭'}")
    print()
    
    client = OpenAI(
        api_key=api_key or "not-needed",
        base_url=f"{base_url}/v1",
        timeout=60.0,
    )
    
    # 测试1: 非流式对话
    print("[测试 1] 非流式对话")
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        *DEFAULT_TEST_CASES[:1]
    ]
    success1 = test_chatCompletion(client, model_name, messages)
    print()
    
    # 测试2: 多轮对话
    print("[测试 2] 多轮对话")
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "最近总是失眠"},
        {"role": "assistant", "content": "失眠确实很让人困扰，请问这种情况持续多久了？"},
        {"role": "user", "content": "大概有两周了"},
    ]
    success2 = test_chatCompletion(client, model_name, messages)
    print()
    
    # 测试3: 流式输出
    if use_streaming:
        print("[测试 3] 流式输出")
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            DEFAULT_TEST_CASES[1]
        ]
        success3 = test_streaming(client, model_name, messages)
        print()
    else:
        success3 = True
    
    # 测试4: 不同 temperature
    print("[测试 4] 高 temperature (创意模式)")
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        DEFAULT_TEST_CASES[2]
    ]
    success4 = test_chatCompletion(client, model_name, messages, temperature=1.0)
    print()
    
    # 总结
    print("=" * 60)
    all_passed = success1 and success2 and success3 and success4
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查上面输出")
    print("=" * 60)
    
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description="MindHealer API 测试")
    parser.add_argument("--url", "-u", default="http://localhost:8000",
                        help="API 服务地址")
    parser.add_argument("--key", "-k", default=None,
                        help="API 密钥（不填则使用默认）")
    parser.add_argument("--model", "-m", default="mindhealer-gemma4-psych",
                        help="模型名称")
    parser.add_argument("--no-stream", action="store_true",
                        help="禁用流式输出测试")
    
    args = parser.parse_args()
    
    return run_tests(
        base_url=args.url,
        api_key=args.key,
        model_name=args.model,
        use_streaming=not args.no_stream,
    )


if __name__ == "__main__":
    sys.exit(main())

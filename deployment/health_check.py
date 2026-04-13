#!/usr/bin/env python3
"""
MindHealer 服务健康检查脚本
用于检查 vLLM/Ollama 部署的服务是否正常运行
"""

import sys
import requests
import argparse
from typing import Optional


def check_vllm(base_url: str, api_key: str, timeout: int = 10) -> bool:
    """检查 vLLM 服务健康状态"""
    
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    # 检查 health 端点
    try:
        resp = requests.get(f"{base_url}/health", headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(f"  ❌ Health 检查失败: HTTP {resp.status_code}")
            return False
        print(f"  ✅ Health 端点正常")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Health 端点无法访问: {e}")
        return False
    
    # 检查模型列表
    try:
        resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=timeout)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            print(f"  ✅ 模型列表获取成功，共 {len(models)} 个模型")
            for m in models:
                print(f"     - {m.get('id', 'unknown')}")
        else:
            print(f"  ⚠️  模型列表请求失败: HTTP {resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  模型列表请求失败: {e}")
    
    # 检查推理能力（发一个简单请求）
    try:
        payload = {
            "model": "mindhealer-gemma4-psych",
            "messages": [
                {"role": "user", "content": "你好"}
            ],
            "max_tokens": 10,
            "stream": False
        }
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=timeout
        )
        if resp.status_code == 200:
            print(f"  ✅ 推理测试成功")
            return True
        else:
            print(f"  ❌ 推理测试失败: HTTP {resp.status_code}")
            print(f"     {resp.text[:200]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 推理测试失败: {e}")
        return False


def check_ollama(base_url: str, timeout: int = 10) -> bool:
    """检查 Ollama 服务健康状态"""
    
    # 检查 tags 端点
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"  ✅ Ollama 连接正常，共 {len(models)} 个模型")
            for m in models:
                print(f"     - {m.get('name', 'unknown')}")
        else:
            print(f"  ⚠️  Ollama API 请求失败: HTTP {resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Ollama 无法访问: {e}")
        return False
    
    # 简单的生成测试
    try:
        payload = {
            "model": "mindhealer",
            "messages": [
                {"role": "user", "content": "你好"}
            ],
            "stream": False
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        if resp.status_code == 200:
            print(f"  ✅ Ollama 推理测试成功")
            return True
        else:
            print(f"  ❌ Ollama 推理测试失败: HTTP {resp.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Ollama 推理测试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MindHealer 服务健康检查")
    parser.add_argument("--type", "-t", choices=["vllm", "ollama"], default="vllm",
                        help="服务类型")
    parser.add_argument("--url", "-u", default="http://localhost:8000",
                        help="服务地址")
    parser.add_argument("--api-key", "-k", default="mindhealer-secret-key-2024",
                        help="API 密钥（vLLM 用）")
    parser.add_argument("--timeout", default=10, type=int,
                        help="请求超时时间（秒）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"MindHealer 服务健康检查")
    print(f"=" * 60)
    print(f"服务类型: {args.type}")
    print(f"服务地址: {args.url}")
    print()
    
    if args.type == "vllm":
        success = check_vllm(args.url, args.api_key, args.timeout)
    else:
        success = check_ollama(args.url, args.timeout)
    
    print()
    print("=" * 60)
    if success:
        print("✅ 服务运行正常")
    else:
        print("❌ 服务可能存在问题，请检查日志")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
MindHealer 模型验证脚本
验证合并后的模型能否正常加载和推理
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./outputs/merged/gemma4-psych-consultant"
    
    print(f"=" * 60)
    print(f"MindHealer 模型验证")
    print(f"=" * 60)
    print(f"模型路径: {model_path}")
    print()
    
    # 1. 检查 CUDA
    print("[1/4] 检查 CUDA 环境...")
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA 不可用，将使用 CPU（会很慢）")
        device = "cpu"
    else:
        print(f"  ✅ CUDA 可用，设备数量: {torch.cuda.device_count()}")
        print(f"  ✅ 当前设备: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    print()
    
    # 2. 加载 tokenizer
    print("[2/4] 加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"  ✅ Tokenizer 加载成功")
        print(f"  ✅ Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"  ❌ Tokenizer 加载失败: {e}")
        return 1
    print()
    
    # 3. 加载模型
    print("[3/4] 加载模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print(f"  ✅ 模型加载成功")
        print(f"  ✅ 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        model.eval()
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return 1
    print()
    
    # 4. 推理测试
    print("[4/4] 推理测试...")
    test_prompts = [
        "我最近总是失眠，怎么办？",
        "工作压力很大，感到焦虑怎么办？",
    ]
    
    system_prompt = "你是一位专业、温暖、有同理心的心理咨询师。"
    
    for i, user_input in enumerate(test_prompts):
        print(f"\n  测试 {i+1}: {user_input}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  回复: {response[:200]}...")
    
    print()
    print("=" * 60)
    print("✅ 模型验证通过！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

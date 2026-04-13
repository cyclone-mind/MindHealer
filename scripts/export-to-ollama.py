"""
scripts/export-to-ollama.py
把合并后的模型转换成 Ollama 需要的 GGUF 格式

用法:
  python scripts/export-to-ollama.py --input ./outputs/merged/gemma4-psych-consultant --output ./mindhealer.gguf
"""

import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def export_to_gguf(model_path, output_path, quantization_type="Q4_K_M"):
    """
    用 llama.cpp 的 convert_hf_to_gguf.py 转换格式
    """
    import subprocess

    print(f">>> 加载模型: {model_path}")

    # 先确认模型能加载
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("   Tokenizer 加载成功")

    # 调用 llama.cpp 的转换脚本
    # 假设 llama.cpp 已经在系统里
    llama_cpp_path = Path(__file__).parent.parent / "llama.cpp"
    if not llama_cpp_path.exists():
        print("克隆 llama.cpp...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp_path)])

    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    print(f">>> 转换为 GGUF 格式...")
    cmd = [
        "python", str(convert_script),
        str(model_path),
        "--outfile", str(output_path),
        "--outtype", quantization_type.lower(),
    ]

    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✅ 转换完成: {output_path}")
    else:
        print(f"❌ 转换失败: {result.returncode}")

    return result.returncode == 0


def generate_ollama_modelfile(model_path, output_path="Modelfile"):
    """
    生成 Ollama 的 Modelfile
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    template = tokenizer.chat_template or (
        "{{ if .System }}\n{{ .System }}\n{{ end }}"
        "{{ range .Messages }}\n{{ if eq .Role \"user\" }}"
        "\nUSER: {{ .Content }}\n"
        "{{ else }}"
        "\nASSISTANT: {{ .Content }}\n"
        "{{ end }}\n{{ end }}"
        "{{ if .Response }}\nASSISTANT: {{ .Response }}\n{{ end }}"
    )

    modelfile_content = f"""FROM ./mindhealer-gguf.gguf

# Template 来自 tokenizer.chat_template
TEMPLATE \"\"\"{template}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop <end_of_turn>

SYSTEM """
你是一位专业、温暖、有同理心的心理咨询师。你擅长理性情绪行为疗法（REBT），
能够通过提问引导来访者自我探索。你的回复温和、专业、富有共情。
"""
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"✅ Modelfile 生成: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出模型为 Ollama 格式")
    parser.add_argument("--input", type=str, required=True, help="合并后的模型路径")
    parser.add_argument("--output", type=str, default="./mindhealer.gguf", help="输出 GGUF 路径")
    parser.add_argument("--quant", type=str, default="Q4_K_M", help="量化类型: Q4_K_M, Q5_K_M, Q8_0")

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"错误：找不到模型目录 {args.input}")
        exit(1)

    # 转换
    ok = export_to_gguf(args.input, args.output, args.quant)

    # 生成 Modelfile
    if ok:
        generate_ollama_modelfile(args.input)

    print("\n下一步:")
    print("  ollama create mindhealer -f Modelfile")
    print("  ollama run mindhealer")

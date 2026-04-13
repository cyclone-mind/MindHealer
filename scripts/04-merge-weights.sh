#!/bin/bash
#
# 04-merge-weights.sh
# 把 LoRA 权重合并回基座模型，生成可直接部署的完整模型
#

set -e

echo "=========================================="
echo "合并 LoRA 权重到基座模型"
echo "=========================================="

LORA_DIR="./outputs/gemma4-qlora-psych"
OUTPUT_DIR="./outputs/merged/gemma4-psych-consultant"
BASE_MODEL="google/gemma-4-E4B-it"

if [ ! -d "$LORA_DIR" ]; then
    echo "错误：找不到 LoRA 权重目录 $LORA_DIR"
    echo "请先运行: bash scripts/03-train-qlora.sh"
    exit 1
fi

cd LLaMA-Factory

mkdir -p ../outputs/merged

echo "开始合并..."
echo "  LoRA 目录: $LORA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 合并权重
llamafactory-cli export \
    --model "$BASE_MODEL" \
    --adapter "$LORA_DIR" \
    --template gemma \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "合并完成！"
echo "合并后的模型在: $OUTPUT_DIR"
echo ""
echo "下一步部署："
echo "  vLLM:  cd deployment/vllm && bash start-vllm.sh"
echo "  或:    python inference/chat.py"
echo "=========================================="

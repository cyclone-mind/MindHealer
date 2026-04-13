#!/bin/bash
#
# 03-train-qlora.sh
# QLoRA 微调训练脚本
#

set -e

echo "=========================================="
echo "MindHealer QLoRA 训练"
echo "基座: google/gemma-4-E4B-it"
echo "=========================================="

# 检查 LLaMA-Factory 是否存在
if [ ! -d "./LLaMA-Factory" ]; then
    echo "LLaMA-Factory 未找到，正在克隆..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi

cd LLaMA-Factory

# 创建输出目录
mkdir -p ../outputs/gemma4-qlora-psych

# 开始训练
# --stage sft: supervised fine-tuning
# --model google/gemma-4-E4B-it: 从 ModelScope 拉模型
# --dataset psych_counseling: 使用预处理后的数据
# --template gemma: Gemma 模板
# --lora_rank 8: LoRA 秩
# --lora_alpha 16: LoRA alpha
# --quantization_bit 4: 4bit 量化
llamafactory-cli train \
    --stage sft \
    --model google/gemma-4-E4B-it \
    --template gemma \
    --dataset psych_counseling \
    --dataset_dir ../data/processed \
    --output_dir ../outputs/gemma4-qlora-psych \
    --deepspeed configs/deepspeed/ds_zereo2.json \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,v_proj \
    --quantization_bit 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bf16 true \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --warmup_ratio 0.03 \
    --ddp_timeout 180000000

echo ""
echo "=========================================="
echo "训练完成！"
echo "LoRA 权重保存在: ../outputs/gemma4-qlora-psych"
echo ""
echo "下一步: bash scripts/04-merge-weights.sh"
echo "=========================================="

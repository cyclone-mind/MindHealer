#!/bin/bash
#
# deployment/vllm/start-vllm.sh
# 使用 vLLM 启动 MindHealer 模型
#

set -e

MODEL_PATH=${1:-"./outputs/merged/gemma4-psych-consultant"}
PORT=${2:-8000}
API_KEY=${3:-"mindhealer-secret-key-2024"}

echo "=========================================="
echo "vLLM 启动脚本"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "API Key: $API_KEY"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误：找不到模型目录 $MODEL_PATH"
    echo "请先运行: bash scripts/04-merge-weights.sh"
    exit 1
fi

# 检查 vLLM 是否安装
if ! python -c "import vllm" 2>/dev/null; then
    echo "安装 vLLM..."
    pip install vllm
fi

echo "启动 vLLM 服务..."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name mindhealer-gemma4-psych \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --gpu-memory-utilization 0.88 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --trust-remote-code

echo ""
echo "=========================================="
echo "vLLM 已启动！"
echo "API 文档: http://localhost:$PORT/docs"
echo "=========================================="

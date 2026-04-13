#!/bin/bash
#
# 01-download-data.sh
# 下载两个数据集：self-cognition 和 PsyDTCorpus
# ModelScope SDK 方式

set -e

DATA_ROOT="./data/raw"
mkdir -p "$DATA_ROOT"

echo "=== 开始下载数据集 ==="

# 检查是否安装 modelscope
if ! python -c "import modelscope" 2>/dev/null; then
    echo "安装 modelscope..."
    pip install modelscope
fi

echo ""
echo ">>> 1/2 下载 swift/self-cognition（角色认同数据）"
python -c "
from modelscope import snapshot_download
import os

cache_dir = '$DATA_ROOT/self-cognition'
os.makedirs(cache_dir, exist_ok=True)

snapshot_download(
    'swift/self-cognition',
    cache_dir=cache_dir
)
print('self-cognition 下载完成')
"

echo ""
echo ">>> 2/2 下载 YIRONGCHEN/PsyDTCorpus（心理咨询对话数据）"
python -c "
from modelscope import snapshot_download
import os

cache_dir = '$DATA_ROOT/PsyDTCorpus'
os.makedirs(cache_dir, exist_ok=True)

snapshot_download(
    'YIRONGCHEN/PsyDTCorpus',
    cache_dir=cache_dir
)
print('PsyDTCorpus 下载完成')
"

echo ""
echo "=== 数据下载完成 ==="
echo "数据保存在: $DATA_ROOT/"
ls -lh "$DATA_ROOT/"

# 顺便下载基座模型（可选，训练脚本会自动从 ModelScope 拉）
echo ""
echo ">>> 可选：下载基座模型 google/gemma-4-E4B-it"
read -p "是否下载基座模型？（需要约 16GB 空间，y/N）" confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    python -c "
    from modelscope import snapshot_download
    import os

    cache_dir = './models/google-gemma-4-E4B-it'
    os.makedirs(cache_dir, exist_ok=True)

    snapshot_download(
        'google/gemma-4-E4B-it',
        cache_dir=cache_dir
    )
    print('基座模型下载完成')
    "
    echo "模型保存在: ./models/google-gemma-4-E4B-it"
fi

echo ""
echo "✅ 全部完成！"

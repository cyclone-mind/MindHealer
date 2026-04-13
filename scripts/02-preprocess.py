"""
02-preprocess.py
把 self-cognition 和 PsyDTCorpus 处理成 LLaMA-Factory 可用的格式。

self-cognition 负责注入身份认同（"你是一个心理咨询师..."）
PsyDTCorpus 负责对话能力

输出格式：LLaMA-Factory 的 dataset JSONL
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 第一步：加载 self-cognition，找到身份设定
# ============================================================

print(">>> 加载 self-cognition 数据...")
sc_path = RAW_DIR / "self-cognition"
sc_file = None
for f in os.listdir(sc_path):
    if f.endswith(".json") or f.endswith(".jsonl"):
        sc_file = sc_path / f
        break

if not sc_file:
    # 尝试从子目录找
    for root, dirs, files in os.walk(sc_path):
        for f in files:
            if f.endswith(".json") or f.endswith(".jsonl"):
                sc_file = Path(root) / f
                break

self_cog_data = []
with open(sc_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            self_cog_data.append(json.loads(line))

print(f"   self-cognition 条目数: {len(self_cog_data)}")

# 随便找一条看看结构
sample = self_cog_data[0]
print(f"   示例 keys: {list(sample.keys())}")
# 通常是 instruction/output 格式，或者 messages 格式

# 从 self-cognition 提取身份设定模板
# 典型字段：instruction="请介绍一下你自己"，output="我是一个心理咨询师..."
# 我们把 output 里的身份描述抽出来，做成新的 system prompt

identity_samples = []
for item in self_cog_data:
    # 兼容不同格式
    if "output" in item:
        identity_samples.append({
            "role": "assistant",
            "content": item["output"]
        })
    elif "messages" in item:
        for msg in item["messages"]:
            if msg.get("role") == "assistant":
                identity_samples.append(msg)
                break

# 取有"心理咨询师"相关内容的作为 identity
psych_identity = None
for s in identity_samples:
    content = s["content"]
    if "心理" in content or "咨询师" in content or "counselor" in content.lower():
        psych_identity = content
        break

# 如果找不到，用第一条
if not psych_identity:
    psych_identity = identity_samples[0]["content"]

print(f"   提取到身份设定: {psych_identity[:80]}...")

# ============================================================
# 第二步：加载 PsyDTCorpus
# ============================================================

print("\n>>> 加载 PsyDTCorpus...")

psy_path = RAW_DIR / "PsyDTCorpus"

# PsyDTCorpus 有两个文件：训练集和测试集
train_file = None
for f in os.listdir(psy_path):
    if "train" in f and (f.endswith(".json") or f.endswith(".jsonl")):
        train_file = psy_path / f
        break

if not train_file:
    for root, dirs, files in os.walk(psy_path):
        for f in files:
            if "train" in f and (f.endswith(".json") or f.endswith(".jsonl")):
                train_file = Path(root) / f
                break

psy_train_data = []
with open(train_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            psy_train_data.append(json.loads(line))

print(f"   PsyDTCorpus 训练集条数: {len(psy_train_data)}")

# 取一条看结构
sample_psy = psy_train_data[0]
print(f"   示例 keys: {list(sample_psy.keys())}")
# 结构: id, normalizedTag, messages: [{role, content}, ...]

# ============================================================
# 第三步：把身份设定注入到 PsyDTCorpus 的 system prompt 里
# ============================================================

print("\n>>> 注入身份设定到 PsyDTCorpus...")

def inject_identity(messages, identity):
    """
    把 identity 注入到 messages 的 system prompt 里。
    如果原本有 system prompt，就拼接在后面。
    """
    result = []
    identity_text = identity.strip()

    for msg in messages:
        if msg["role"] == "system":
            # 原本有 system，拼接在后面
            original = msg["content"].strip()
            new_system = original + "\n\n" + identity_text
            result.append({"role": "system", "content": new_system})
        else:
            result.append(msg)

    # 如果没有 system，第一条插入
    if not any(m["role"] == "system" for m in messages):
        result.insert(0, {"role": "system", "content": identity_text})

    return result

# 处理后的数据
processed_data = []
tag_counts = {}

for item in psy_train_data:
    messages = item["messages"]
    tag = item.get("normalizedTag", "unknown")
    tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # 注入身份
    messages_with_identity = inject_identity(messages, psych_identity)

    processed_data.append({
        "messages": messages_with_identity,
        "category": tag,
    })

print(f"   处理完毕: {len(processed_data)} 条")
print(f"   类别分布: {tag_counts}")

# ============================================================
# 第四步：保存为 LLaMA-Factory 的 JSONL 格式
# ============================================================

print("\n>>> 保存处理后的数据...")

output_file = OUT_DIR / "psych_counseling_train.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in processed_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"   保存到: {output_file}")
print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

# ============================================================
# 第五步：生成 dataset_info.json（LLaMA-Factory 需要）
# ============================================================

dataset_info = {
    "psych_counseling": {
        "file_name": "psych_counseling_train.jsonl",
        "formatting": "sharegpt",
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}

info_file = OUT_DIR / "dataset_info.json"
with open(info_file, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print(f"   dataset_info.json 已生成")

# ============================================================
# 第六步：顺便处理测试集（如果有的话）
# ============================================================

test_file = None
for f in os.listdir(psy_path):
    if "test" in f and (f.endswith(".json") or f.endswith(".jsonl")):
        test_file = psy_path / f
        break

if test_file:
    print("\n>>> 处理测试集...")

    psy_test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                psy_test_data.append(json.loads(line))

    print(f"   测试集条数: {len(psy_test_data)}")

    processed_test = []
    for item in psy_test_data:
        messages = item["messages"]
        tag = item.get("normalizedTag", "unknown")
        messages_with_identity = inject_identity(messages, psych_identity)
        processed_test.append({
            "messages": messages_with_identity,
            "category": tag,
        })

    test_output = OUT_DIR / "psych_counseling_test.jsonl"
    with open(test_output, "w", encoding="utf-8") as f:
        for item in processed_test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   测试集保存到: {test_output}")
    print(f"   文件大小: {test_output.stat().st_size / 1024 / 1024:.2f} MB")

    # 更新 dataset_info
    dataset_info["psych_counseling"]["test_file"] = "psych_counseling_test.jsonl"
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("\n✅ 数据预处理完成！")
print(f"\n接下来可以运行: bash scripts/03-train-qlora.sh")

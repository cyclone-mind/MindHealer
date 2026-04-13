# MindHealer 故障排查指南

常见问题汇总，按症状分类。

---

## 目录

- [训练相关](#训练相关)
- [部署相关](#部署相关)
- [推理相关](#推理相关)
- [其他问题](#其他问题)

---

## 训练相关

### 训练时显存溢出（OOM）

**症状**：GPU 显存不够，进程被 kill

**解法**（按优先级尝试）：

1. 减小 batch size：
   ```yaml
   per_device_train_batch_size: 1  # 从 2 降到 1
   ```

2. 增大 gradient accumulation：
   ```yaml
   gradient_accumulation_steps: 16  # 从 8 增到 16
   ```

3. 开启梯度检查点：
   ```yaml
   gradient_checkpointing: true
   ```

4. 降低模型精度（最后手段）：
   ```yaml
   fp16: true
   bf16: false
   ```

---

### LLaMA-Factory 报 `KeyError: 'gemma'`

**症状**：训练启动时报 tokenizer 相关错误

**原因**：LLaMA-Factory 对 Gemma 的支持不完善

**解法**：

1. 确保 transformers 版本够新：
   ```bash
   pip install transformers>=4.43.0
   ```

2. 启动训练时指定模板：
   ```bash
   llamafactory-cli train examples/train_full/qwen.yaml --template gemma
   ```

3. 如果还是不行，换 Qwen 基座（见下文）

---

### 训练loss正常但生成效果差

**症状**：loss 下降正常，但模型输出乱码或无意义

**可能原因**：
- system prompt 太长导致 context 被污染
- 数据质量不行
- epoch 不够

**解法**：
1. 减少 system prompt 长度
2. 检查数据清洗是否到位
3. 增加训练 epoch

---

## 部署相关

### vLLM 启动报 tokenizer 错误

**症状**：`KeyError: 'gemma'` 或 `OSError`

**解法**：启动时加 `--trust-remote-code`

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./outputs/merged/gemma4-psych-consultant \
    --trust-remote-code \
    ...
```

---

### vLLM 显存不够（OOM）

**症状**：启动时报 `CUDA out of memory`

**解法**：

1. 降低显存占用：
   ```bash
   --gpu-memory-utilization 0.75  # 从 0.88 降到 0.75
   ```

2. 减小上下文长度：
   ```bash
   --max-model-len 4096  # 从 8192 降到 4096
   ```

3. 单卡改多卡（如果有）：
   ```bash
   --tensor-parallel-size 2
   ```

---

### 服务启动后很快挂掉

**排查步骤**：

1. 检查显存：
   ```bash
   nvidia-smi
   ```
   是不是被其他进程占用了

2. 检查日志：
   看看具体是什么错误

3. 检查并行请求数：
   是不是同时开了太多请求导致 OOM

---

### Ollama 模型加载失败

**症状**：`Error: model requires more memory`

**解法**：

1. 减小上下文窗口：
   编辑 Modelfile，加一行：
   ```
   PARAMETER num_ctx 2048
   ```

2. 换用量化版本

---

## 推理相关

### 生成结果乱码/格式错乱

**症状**：输出中文字符乱码或 JSON 格式错乱

**原因**：Gemma 的 chat template 和 Llama 不一样

**解法**：

1. 确保用 `apply_chat_template` 构建 prompt
2. 启动 vLLM 时加 `--trust-remote-code`
3. 检查 tokenizer_config.json 的 chat_template 配置

---

### 生成速度很慢

**症状**：first token 要等很久，或者生成很慢

**解法**：

1. 使用 vLLM 部署（比原生 transformers 快 10x+）
2. 换用更快的基座模型（如 Qwen2.5-7B）
3. 减小 max_model_len

---

### 回复太短/太长/重复

**症状**：模型要么只说一个字，要么啰嗦个没完，要么车轱辘话

**解法**：

1. 调整 temperature：
   - 太短 → 降低 temperature（0.3-0.5）
   - 太长/重复 → 提高 temperature（0.7-1.0）+ repetition_penalty

2. 调整 max_tokens

3. 优化 system prompt，明确要求输出长度

---

## 其他问题

### ModelScope 下载慢/失败

**症状**：下载数据集速度感人，或者超时

**备选方案**：

```bash
# 用 git-lfs 直接 clone
cd data/raw
git lfs install
git clone https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus.git
```

---

### 合并权重后模型很大

**原因**：合并后的模型包含完整权重，加上 optimizer 等中间状态

**解法**：

1. 只合并权重（不包含优化器）：
   ```bash
   python scripts/04-merge-weights.sh
   ```

2. 使用 `--merge_weights_only` 模式

3. 考虑使用 GGUF 格式压缩

---

### 找不到 PyTorch 或 CUDA

**症状**：`ModuleNotFoundError: No module named 'torch'`

**解法**：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

（待补充更多问题）

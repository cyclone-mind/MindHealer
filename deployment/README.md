# MindHealer 部署指南

本文档详细介绍 MindHealer 模型的多种部署方式，推荐生产环境使用 vLLM，本地快速尝鲜可用 Ollama 或 Gradio。

---

## 目录

- [环境要求](#环境要求)
- [vLLM 部署（推荐）](#vllm-部署推荐)
- [Ollama 部署](#ollama-部署)
- [Gradio 网页（演示用）](#gradio-网页演示用)
- [API 调用示例](#api-调用示例)
- [常见问题](#常见问题)

---

## 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 16GB 显存 | 24GB+（3090/4090/A10/A100） |
| 内存 | 16GB | 32GB+ |
| 硬盘 | 50GB | 100GB+ |
| CUDA | 11.8+ | 12.1+ |
| Python | 3.10 | 3.10 |

确认 CUDA 可用：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## vLLM 部署（推荐）

vLLM 支持 PagedAttention、投机解码等高级功能，吞吐量和显存利用率都比 naive 实现高很多。生产环境必备。

### 步骤 1：安装 vLLM

```bash
pip install vllm
```

或者从源码编译（需要 GPU 驱动支持）：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 步骤 2：准备模型

确保合并后的模型在本地：

```bash
# 默认路径：./outputs/merged/gemma4-psych-consultant
# 如果模型在其他位置，启动时通过参数传入
ls ./outputs/merged/gemma4-psych-consultant/
```

### 步骤 3：启动服务

```bash
cd deployment/vllm
bash start-vllm.sh
```

也可以手动指定参数：

```bash
MODEL_PATH=./outputs/merged/gemma4-psych-consultant
PORT=8000
API_KEY=mindhealer-secret-key-2024

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name mindhealer-gemma4-psych \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --gpu-memory-utilization 0.88 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --trust-remote-code
```

### 启动参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | ./outputs/merged/gemma4-psych-consultant | 模型路径 |
| `--served-model-name` | mindhealer-gemma4-psych | API 中使用的模型名 |
| `--port` | 8000 | HTTP 端口 |
| `--api-key` | mindhealer-secret-key-2024 | API 密钥 |
| `--gpu-memory-utilization` | 0.88 | GPU 显存占用比例 |
| `--max-model-len` | 8192 | 最大上下文长度 |
| `--tensor-parallel-size` | 1 | 多卡并行数（单卡写1，多卡按实际填写） |
| `--trust-remote-code` | - | 允许执行模型自定义代码 |

### 步骤 4：验证服务

```bash
# 检查服务是否启动
curl http://localhost:8000/health

# 查看 API 文档
# 浏览器打开 http://localhost:8000/docs
```

### 停止服务

```bash
# 查找进程
ps aux | grep vllm

# 杀掉进程
kill <PID>
```

---

## Ollama 部署

Ollama 开箱即用，本地简单部署首选，但吞吐量不如 vLLM。

### 步骤 1：安装 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 步骤 2：导出模型为 Ollama 格式

```bash
python scripts/export-to-ollama.py
```

这会在当前目录生成 `Modelfile` 和 GGUF 格式的模型文件。

### 步骤 3：创建并运行模型

```bash
ollama create mindhealer -f Modelfile
ollama run mindhealer
```

### API 调用（OpenAI 兼容）

```python
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

response = client.chat.completions.create(
    model="mindhealer",
    messages=[
        {"role": "system", "content": "你是一位专业、温暖、有同理心的心理咨询师。"},
        {"role": "user", "content": "最近总是失眠，是怎么回事？"}
    ]
)
print(response.choices[0].message.content)
```

---

## Gradio 网页（演示用）

适合在没有公网访问条件时本地演示，不需要额外部署服务。

```bash
python inference/web_demo.py
```

启动后访问 `http://localhost:7860`（或终端显示的地址）即可使用网页界面。

---

## API 调用示例

### Python（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(
    api_key="mindhealer-secret-key-2024",
    base_url="http://localhost:8000/v1",
)

# 单轮对话
response = client.chat.completions.create(
    model="mindhealer-gemma4-psych",
    messages=[
        {
            "role": "system",
            "content": "你是一位专业、温暖、有同理心的心理咨询师。"
        },
        {
            "role": "user",
            "content": "最近工作压力很大，总是很焦虑怎么办？"
        }
    ],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mindhealer-secret-key-2024" \
  -d '{
    "model": "mindhealer-gemma4-psych",
    "messages": [
      {"role": "system", "content": "你是一位专业、温暖、有同理心的心理咨询师。"},
      {"role": "user", "content": "最近总是失眠，是怎么回事？"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### 流式输出

```python
stream = client.chat.completions.create(
    model="mindhealer-gemma4-psych",
    messages=[
        {"role": "system", "content": "你是一位专业、温暖、有同理心的心理咨询师。"},
        {"role": "user", "content": "最近总是失眠，是怎么回事？"}
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 常见问题

### Q: vLLM 启动报 tokenizer 错误

**症状**：`KeyError: 'gemma'` 或 tokenizer 加载失败

**解法**：启动时加 `--trust-remote-code`

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    ...
```

### Q: 显存不够（OOM）

**解法**：
1. 降低 `--gpu-memory-utilization`（从 0.88 降到 0.75）
2. 减小 `--max-model-len`（从 8192 降到 4096）
3. 如果是多卡，改 `--tensor-parallel-size`

### Q: 生成结果乱码或格式错乱

**原因**：Gemma 的 chat template 和标准 Llama 不一样

**解法**：检查模型的 `tokenizer_config.json` 里的 `chat_template` 是否正确配置。vLLM 启动时加 `--trust-remote-code` 通常能解决。

### Q: Gemma 在 vLLM 上生成质量不如 transformers 原生

**原因**：vLLM 对 Gemma 的某些优化可能导致生成风格略有差异

**解法**：换用 Ollama 部署，或者用 `inference/chat.py` 直接用 transformers 推理（虽然慢一点但更稳定）。

### Q: 服务启动后很快挂掉

**检查**：
1. 显存是否被其他进程占用：`nvidia-smi`
2. 是不是开了太多并行请求导致 OOM
3. 看日志具体报错信息

---

## 部署方式对比

| 方案 | 吞吐量 | 显存效率 | 部署难度 | 适用场景 |
|------|--------|----------|----------|----------|
| **vLLM** | ★★★★★ | ★★★★★ | 中等 | 生产环境、公有云 |
| **Ollama** | ★★★ | ★★★ | 简单 | 本地快速尝鲜 |
| **Gradio** | ★★ | ★★ | 最简单 | 演示、无公网访问 |
| **transformers** | ★★ | ★★ | 简单 | 调试、debug |

生产环境推荐 **vLLM**，本地开发推荐 **Ollama** 或 **Gradio**。

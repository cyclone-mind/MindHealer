# MindHealer

中文心理咨询大模型。基于 Google Gemma-4-E4B-it 微调，结合心理咨询师数字孪生对话数据，打造具有共情能力、专业咨询技能的心理健康对话模型。

## 项目背景

做这个项目的原因很简单：市面上的心理咨询工具，要么太机械（只会模板回复），要么太贵（GPT-4 套壳），要么就是缺乏真正的共情能力。

SoulChat2.0 的论文发到 ACL 2025 给了我很大启发——用少量真实案例就能构建出具有特定咨询师风格的数字孪生，这个思路很巧妙。所以我决定基于他们的数据集，自己动手微调一个心理咨询模型。

主要参考：
- [PsyDTCorpus](https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus) — 心理咨询师数字孪生对话数据（ACL 2025）
- [swift/self-cognition](https://modelscope.cn/datasets/swift/self-cognition) — 角色认同数据

## 模型信息

| 项目 | 内容 |
|------|------|
| 基座模型 | google/gemma-4-E4B-it（ModelScope） |
| 参数量 | 8B |
| 微调方式 | QLoRA（4bit NF4 + LoRA） |
| 框架 | LLaMA-Factory |
| 协议 | apache-2.0 |

选 Gemma-4 而不是 Qwen2.5 的原因：Gemma-4 新出，据说在推理和指令遵循上有提升，而且开源协议比 Meta 的 Llama 宽松（apache-2.0 vs Llama 3.1），商用限制更少。不过如果跑不通，LLaMA-Factory 也支持 Qwen2.5，换基座只需要改个模型 ID。

## 快速开始

### 环境要求

- GPU：24GB+ 显存（3090/4090/A10/A100 都行）
- 内存：32GB+
- 硬盘：100GB+（模型约 16GB + 数据约 2GB）

### 1. 克隆项目

```bash
git clone https://github.com/cyclone-mind/MindHealer.git
cd MindHealer
```

### 2. 安装依赖

```bash
conda create -n mindhealer python=3.10
conda activate mindhealer

# 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
cd ..

# 或者用 ms-swift（可选）
# pip install ms-swift -U
```

### 3. 下载数据

```bash
bash scripts/01-download-data.sh
```

这会自动下载两个数据集到 `data/raw/` 目录。

### 4. 数据预处理

```bash
python scripts/02-preprocess.py
```

处理后的数据会保存到 `data/processed/`。

### 5. 启动训练

```bash
bash scripts/03-train-qlora.sh
```

训练大约需要 8-12 小时（取决于 GPU），Loss 降到 0.8 以下可以提前停止。

### 6. 合并权重

```bash
bash scripts/04-merge-weights.sh
```

合并后的模型保存到 `outputs/merged/`。

### 7. vLLM 部署

```bash
cd deployment/vllm
bash start-vllm.sh
```

默认端口 8000，启动后访问 http://localhost:8000/docs 查看 API 文档。

## 目录结构

```
MindHealer/
├── data/
│   ├── raw/                    # 原始数据
│   │   ├── self-cognition/     # 角色认同数据
│   │   └── PsyDTCorpus/        # 心理咨询对话数据
│   └── processed/              # 处理后的数据
├── scripts/
│   ├── 01-download-data.sh     # 下载数据
│   ├── 02-preprocess.py        # 数据预处理
│   ├── 03-train-qlora.sh       # QLoRA 训练
│   └── 04-merge-weights.sh     # 合并权重
├── configs/
│   └── qlora/
│       └── gemma4-qlora.yaml   # QLoRA 训练配置
├── inference/
│   ├── chat.py                 # 命令行对话
│   └── web_demo.py             # Gradio 网页
├── deployment/
│   └── vllm/
│       ├── start-vllm.sh       # 启动脚本
│       └── openai-api.py       # API 调用示例
├── outputs/                     # 训练输出
└── README.md
```

## 数据集说明

### swift/self-cognition

让模型知道自己是"心理咨询师"的身份认同数据。包含中英文版本，大概 5000 条左右。格式是 instruction-response，模型通过这条数据学会"我是一个心理咨询师"这个身份设定。

### YIRONGCHEN/PsyDTCorpus

心理咨询师数字孪生数据，4760 条多轮对话，总共 90,365 轮，平均每轮对话 18 轮。数据质量很高，涵盖了婚恋、职场、家庭、自我成长等多个咨询场景。system prompt 里包含了 REBT（理性情绪行为疗法）的完整技术框架，咨询风格温柔且专业。

训练时会先把 self-cognition 的身份设定注入到 PsyDTCorpus 的 system prompt 里，然后再喂给模型学习。

## 已知问题

1. **Gemma tokenizer 兼容性**：Gemma 用的是 Google 自家的 tokenizer，和 LLaMA-Factory 的某些功能可能不完全兼容。如果遇到问题，优先换 Qwen2.5-7B-Instruct 作为基座（配置里改一行就行）。

2. **中文能力**：Gemma 系列原生中文能力不如 Qwen 强，如果发现中文回复质量不行，果断换基座。

3. **QLoRA vs 全量微调**：这个项目用的是 QLoRA（省显存），但 PsyDTCorpus 原论文用的是全量微调（8× A100）。如果你的目标是最高效果，建议有条件还是全量微调，只需要改一下配置文件里的 `finetuning_type: full`。

## 如果你有更好的 GPU

### 全量微调（需要 8× A100 80GB）

```bash
bash scripts/05-train-full.sh
```

### 多卡 QLoRA

修改 `configs/qlora/gemma4-qlora.yaml` 里的：

```yaml
device_map: auto  # 改成多卡
```

然后：

```bash
llamafactory-cli train configs/qlora/gemma4-qlora.yaml
```

## 部署方式

### vLLM（推荐，生产环境用）

```bash
cd deployment/vllm
bash start-vllm.sh
```

### Ollama（本地简单部署）

```bash
# 先导出模型为 Ollama 格式
python scripts/export-to-ollama.py

# 然后
ollama create mindhealer -f Modelfile
ollama run mindhealer
```

### Gradio 网页（演示用）

```bash
python inference/web_demo.py
```

## License

基于 apache-2.0 协议开源。基座模型遵循 Google Gemma Terms of Use。

## 致谢

- 数据集来源：[SoulChat2.0 / PsyDTCorpus](https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus)（华南理工大学，ACL 2025）
- 框架：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- 基座：[Google Gemma-4-E4B-it](https://modelscope.cn/models/google/gemma-4-E4B-it)

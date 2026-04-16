# MindHealer 更新日志

所有重要的项目变更都会记录在此。

---

## [进行中] v1.0.0（2026-04-15）

### 训练进度

- ✅ 数据下载脚本完成
- ✅ 数据预处理完成
- ⏳ QLoRA 微调进行中
- ⏳ 权重合并待完成
- ⏳ 部署上线待完成

### 新增

- **模型支持**：QLoRA 微调配置（gemma4-qlora.yaml、qwen2.5-7b-qlora.yaml）
- **训练脚本**：`scripts/02-preprocess.py` — 数据预处理
- **部署文档**：`deployment/README.md` — 完整部署指南
  - vLLM 部署（生产推荐）
  - Ollama 部署（本地快速尝鲜）
  - Gradio 网页演示
- **健康检查**：`deployment/health_check.py` — 服务可用性检测
- **模型验证**：`scripts/validate-model.py` — 合并后模型加载验证
- **API 测试**：`scripts/test-api.py` — OpenAI 兼容 API 测试
- **故障排查**：`docs/troubleshooting.md` — FAQ 文档

### 文档

- `README.md` — 项目介绍、快速开始、模型信息
- `docs/project-notes.md` — 开发笔记、思路记录
- `docs/challenges-and-notes.md` — 踩坑记录

### 项目结构

```
MindHealer/
├── configs/qlora/          # QLoRA 训练配置
│   ├── gemma4-qlora.yaml
│   └── qwen2.5-7b-qlora.yaml
├── deployment/             # 部署相关
│   ├── vllm/              # vLLM 部署
│   ├── ollama/            # Ollama 部署
│   └── health_check.py
├── docs/                  # 文档
├── inference/             # 推理脚本
├── scripts/               # 工具脚本
└── outputs/              # 训练输出
```

### 技术栈

- 基座模型：google/gemma-4-E4B-it（8B, apache-2.0）
- 微调方式：QLoRA（4bit NF4 + LoRA）
- 框架：LLaMA-Factory
- 部署：vLLM / Ollama / Gradio

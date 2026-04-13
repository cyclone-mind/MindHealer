# 踩坑记录

这个文件用来记一些开发过程中遇到的破事，省得下次再踩一遍。

## 2026-04-13

### Gemma tokenizer 和 LLaMA-Factory 兼容问题

LLaMA-Factory 对 Gemma 的支持没有 Qwen 那么完善。Gemma 用的是 Google 自家的 tokenizer，chat template 格式和标准 Llama 不太一样。

**症状**：
训练脚本跑起来后，报 `KeyError: 'gemma'` 或者 tokenizer 相关的错误。

**解法**：
1. 确保 transformers >= 4.43.0
2. 启动训练时加 `--template gemma`
3. 如果还是报错，检查 LLaMA-Factory 的版本，可能需要更新到最新

**备选方案**：
如果折腾了半天还是不行，果断换 Qwen2.5-7B-Instruct 作为基座。配置文件里改一行：
```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
template: qwen
```
其他全部不用动。

### PsyDTCorpus 的 system prompt 太长了

PsyDTCorpus 里的 system prompt 是完整的 REBT 疗法说明，非常长（好几千字）。这导致：

1. context length 容易被撑满
2. 训练时显存压力大
3. 实际上不需要那么长的 system prompt

**我的处理方式**：
把 self-cognition 里"心理咨询师身份认同"的内容拼到 system prompt 前面，然后截断 REBT 那段超长的说明。实际训练时发现，保留前面的身份认同和咨询原则就够了，来访者不会在意你用的是什么具体疗法技术。

### ModelScope 下载速度感人

用 `modelscope snapshot_download` 下数据集的时候，速度不太稳定。有时候几百MB要下半天。

**备选**：
用 git-lfs 直接 clone：
```bash
cd data/raw
git lfs install
git clone https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus.git
```

速度会快很多，前提是网络能通。

---

（待补充）

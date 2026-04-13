# 项目笔记

## 为什么选 Gemma-4

 Gemma-4 是 Google 2026 年新出的模型，31B 那款下载量 527k 说明确实有人在用。8B 的版本对
 硬件要求没那么离谱，24GB 显卡基本能跑起来。

 选它而不是 Qwen2.5 的原因：
 1. apache-2.0 协议，商用限制少
 2. 新出的，理论上对长上下文支持更好（心理咨询往往要聊很多轮）
 3. 纯好奇，想试试 Google 的模型到底行不行

 如果跑崩了就换 Qwen2.5-7B，反正配置里就改一行。

## 数据集踩坑记录

 ### self-cognition 数据格式
 这数据集本质是让模型认清楚"我是谁"。格式大概是：
 ```json
 {"instruction": "请介绍一下你自己", "output": "我是一个心理咨询师..."}
 ```
 实际上 output 里的内容挺杂的，有中文有英文，还有各种花里胡哨的描述。

 真正有用的就是那个"心理咨询师"的身份设定。所以预处理的时候只把包含"心理"或
 "咨询师"关键字的 output 抽出来当 system prompt 用。

 ### PsyDTCorpus 数据格式
 这个数据集质量确实可以。messages 格式是标准的 OpenAI 多轮对话格式：
 ```json
 {
   "id": 0,
   "normalizedTag": "婚恋",
   "messages": [
     {"role": "system", "content": "你是一位精通REBT的心理咨询师..."},
     {"role": "user", "content": "晚上好..."},
     {"role": "assistant", "content": "晚上好..."}
   ]
 }
 ```

 system prompt 巨长，是完整的 REBT 疗法技术说明。但这不是针对"心理咨询师身份"
 的说明，更像是"这个咨询师用什么疗法"。

 所以预处理的时候把 self-cognition 里的身份设定拼到 PsyDTCorpus 的 system prompt 后面。

 ## Gemma tokenizer 的坑

 Gemma 用的是 Google 自家的 SentencePiece tokenizer，和 LLaMA 不一样。

 LLaMA-Factory 的 Gemma 模板是这样的：
 ```
 <start_of_turn>user
 你好<end_of_turn>
 <start_of_turn>model
 你好呀<end_of_turn>
 ```

 注意每个 turn 都有明确的 start/end token。chat_template 在 tokenizer_config.json 里。

 如果 LLaMA-Factory 报 tokenizer 相关的错误，先检查：
 1. transformers 版本够不够新（4.43.0+）
 2. tokenizer_config.json 有没有正确配置 template
 3. Gemma 的 chat_template 格式可能和 Qwen 不一样

 实在不行就换基座，这个项目的目标不是死磕 Gemma，而是验证心理咨询微调这条路
 是走得通的。

 ## 显存估算

 QLoRA 4bit 加载 Gemma-4-8B：
 - 模型权重：~8GB（4bit 量化后）
 - 梯度 + 优化器状态：LoRA 部分很小，~100MB
 -  activations：batch_size=1 的话，~2-3GB

 总共 24GB 显卡够用。但如果跑着跑着爆了，检查一下：
 - 是不是开了 deepspeed 多卡
 - batch_size 是不是设太大了
 - max_length 是不是设太高了（默认 2048 够用）

 ## vLLM 部署的坑

 vLLM 目前对 Gemma 的支持不如 Qwen 稳定。如果遇到：
 - 启动时报 tokenizer 错误：加 --trust-remote-code
 - 生成结果全是乱码：检查 chat_template 对不对
 - 显存不够：降 --gpu-memory-utilization 到 0.8

 如果 vLLM 实在跑不动 Gemma，换成本地 transformers 推理（inference/chat.py 里有备选代码），
 虽然慢一点但肯定能跑。

 ## 相关链接

 - SoulChat2.0 论文: https://aclanthology.org/2025.acl-long.55/
 - PsyDTCorpus: https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus
 - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
 - ms-swift: https://github.com/modelscope/ms-swift
 - Google Gemma-4: https://modelscope.cn/models/google/gemma-4-E4B-it

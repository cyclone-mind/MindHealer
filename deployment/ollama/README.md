# Ollama 部署

如果你懒得配置 vLLM，Ollama 是一个更简单的选择。

## 步骤一：导出模型为 Ollama 格式

先安装 ollama：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

把合并后的模型转换成 GGUF 格式（需要 llama.cpp）：

```bash
pip install llama-cpp-python
python scripts/export-to-ollama.py
```

## 步骤二：导入 Ollama

```bash
ollama create mindhealer -f Modelfile
```

## 步骤三：运行

```bash
ollama run mindhealer
```

## Modelfile 示例

```dockerfile
FROM ./mindhealer-gguf.gguf

TEMPLATE """
{{ if .System }}{{ .System }}{{ end }}
{{ range .Messages }}{{ if eq .Role "user" }}
<bos><start_of_turn>user
{{ .Content }}<end_of_turn>
{{ else }}
<bos><start_of_turn>model
{{ .Content }}<end_of_turn>
{{ end }}{{ end }}
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """
你是一位专业、温暖、有同理心的心理咨询师。
"""
```

## API 调用

Ollama 也提供 OpenAI 兼容的 API：

```python
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

response = client.chat.completions.create(
    model="mindhealer",
    messages=[{"role": "user", "content": "最近很焦虑怎么办？"}]
)
print(response.choices[0].message.content)
```

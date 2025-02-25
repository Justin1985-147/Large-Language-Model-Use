# how to use api to get response from ollama

# 导入OpenAI类，用于与API交互
from openai import OpenAI

modelname = "deepseek-r1:14b" # or modelname = "deepseek-r1:32b"

# Message format
messages = [
    {
        "role": "system",
        "content": "你是AI助手小明，是我的私人助理。我每周固定周三、周五要开会。"
    },
    {
        "role": "user",
        "content": "我哪天有会？"
    },
]

# 初始化OpenAI客户端，设置基础URL为本地Ollama服务器地址，并提供API密钥。
# 本地部署的Ollama的api_key随便写就可以。
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="Liu")
 
 # Call deepseek-chat
chat_completion = client.chat.completions.create(
    model= modelname,
    messages=messages,
    stream=False,
    temperature=0.7,
    response_format={"type": "text"} # or response_format={"type": "json_object"}
)

# Output reply
print(chat_completion.choices[0].message.content)

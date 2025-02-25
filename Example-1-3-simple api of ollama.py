# how to use api to get response from ollama

# you need to install ollama first and run it locally
# you also need to download the model you want to use by ollama

# Import OpenAI class for API interaction
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

# Initialize OpenAI client, set base URL to local Ollama server address and provide API key
# For local Ollama deployment, the api_key can be any value
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

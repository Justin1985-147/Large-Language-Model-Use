# This is a simple test example of calling the deepseek API from Siliconflow.
#——————————————————————————————————————————————————————————————————
# Python environment needs to install python-dotenv openai
# !pip install python-dotenv openai
# !pip install --upgrade openai
# Additionally, you need to configure the .env file, which needs to specify OPENAI_API_KEY and OPENAI_BASE_URL
""" By default, Python reads files using the default encoding of Windows, which is gbk.
However, modern documents are generally in utf-8 format. Therefore, Python needs to be set to utf-8 mode.
You can search for environment variables, then configure the environment variable PYTHONUTF8, with a value of 1. """
#——————————————————————————————————————————————————————————————————

modelname = "deepseek-ai/DeepSeek-V3" # or modelname = "deepseek-ai/DeepSeek-R1"

from openai import OpenAI
import os

# Load .env file into environment variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
api_key = os.getenv("SC_API_KEY")
base_url = os.getenv("SC_API_BASE")

# Initialize OpenAI service.
client = OpenAI(api_key=api_key, base_url=base_url)

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

"""
场景	温度
代码生成/数学解题   	0.0
数据抽取/分析	1.0
通用对话	1.3
翻译	1.3
创意类写作/诗歌创作	1.5
"""
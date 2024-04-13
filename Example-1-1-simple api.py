# This is a simple test example of calling the GPT API.
#——————————————————————————————————————————————————————————————————
# Python environment needs to install python-dotenv openai
# !pip install python-dotenv openai
# !pip install --upgrade openai
# Additionally, you need to configure the .env file, which needs to specify OPENAI_API_KEY and OPENAI_BASE_URL
""" By default, Python reads files using the default encoding of Windows, which is gbk.
However, modern documents are generally in utf-8 format. Therefore, Python needs to be set to utf-8 mode.
You can search for environment variables, then configure the environment variable PYTHONUTF8, with a value of 1. """
#——————————————————————————————————————————————————————————————————

modelname = "gpt-3.5-turbo"

from openai import OpenAI

# Load .env file into environment variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Initialize OpenAI service. It will automatically load OPENAI_API_KEY and OPENAI_BASE_URL from environment variables.
client = OpenAI()

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

# Call GPT-3.5
chat_completion = client.chat.completions.create(
    model= modelname,
    messages=messages
)

# Output reply
print(chat_completion.choices[0].message.content)
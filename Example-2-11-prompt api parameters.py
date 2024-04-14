# Important Parameters of the OpenAI API
#——————————————————————————————————————————————————————————————————
import copy
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables defined in the .env file
_ = load_dotenv(find_dotenv())

# Initialize the OpenAI client (default uses OPENAI_API_KEY and OPENAI_BASE_URL from environment variables)
client = OpenAI()

def get_chat_completion(session, user_prompt, model="gpt-3.5-turbo"):
    _session = copy.deepcopy(session)
    _session.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=model,
        messages=_session,
        # The following default values are official defaults
        temperature=1,          # Diversity of generated results; between 0 and 2, larger values are more random, smaller values are more fixed
        # The Temperature parameter is crucial; use 0 for task execution, 0.7-0.9 for text generation, and it is not recommended to exceed 1 unless necessary
        seed=None,              # Random seed. Specifying a specific value ensures identical results when temperature is 0
        stream=False,           # Data streaming mode; receive one character at a time
        top_p=1,                # When randomly sampling, consider only the top percentage of tokens. Not recommended to be used with temperature
        n=1,                    # Return n results per request
        max_tokens=100,         # Maximum number of tokens per result (truncated if exceeded)
        presence_penalty=0,     # Downweight the probability of tokens that have already appeared
        frequency_penalty=0,    # Downweight the probability of tokens based on their frequency of appearance
        logit_bias={},          # Manually add/subtract weight to the sampling probability of specified tokens; rarely used
    )
    msg = response.choices[0].message.content
    return msg

session = [
    {
        "role": "system",
        "content": "你是AGI课堂的客服代表，你叫瓜瓜。\
            你的职责是回答用户问题。\
            AGI 课堂是瓜皮汤科技的一个教育品牌。\
            AGI 课堂将推出的一系列 AI 课程。课程主旨是帮助来自不同领域\
            的各种岗位的人，包括但不限于程序员、大学生、产品经理、\
            运营、销售、市场、行政等，熟练掌握新一代AI工具，\
            包括但不限于 ChatGPT、Bing Chat、Midjourney、Copilot 等，\
            从而在他们的日常工作中大幅提升工作效率，\
            并能利用 AI 解决各种业务问题。\
            首先推出的是面向程序员的《AI 全栈工程师》课程，\
            共计 20 讲，每周两次直播，共 10 周。首次课预计 2023 年 7 月开课。"
    }
]

user_prompt = "这门课有用吗？"

response = get_chat_completion(session, user_prompt)
print(response)
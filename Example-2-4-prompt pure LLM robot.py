# Implementing a complete customer service functionality solely using OpenAI API
#——————————————————————————————————————————————————————————————————
"""
Multi-turn dialogue requires including conversation history on each turn (yes, it's expensive in terms of tokens).
Interacting with a large language model neither makes it smarter nor dumber.
However, the conversation history data may be used to train such models...
"""
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


def print_json(data):
    """
    Prints the argument. If the argument is structured (e.g., dictionary or list), it is printed as formatted JSON;
    otherwise, the value itself is printed directly.
    """
    if hasattr(data, 'model_dump_json'):
        data = json.loads(data.model_dump_json())

    if isinstance(data, (list, dict)):
        print(json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)


client = OpenAI()

# Define message history, starting with a system message containing non-prompt dialogue content
messages = [
    {
        "role": "system",
        "content": """
你是一个手机流量套餐的客服代表，你叫小瓜。可以帮助用户选择最合适的流量套餐产品。可以选择的套餐包括：
经济套餐，月费50元，10G流量；
畅游套餐，月费180元，100G流量；
无限套餐，月费300元，1000G流量；
校园套餐，月费150元，200G流量，仅限在校生。
"""
    }
]


def get_completion(prompt, model="gpt-3.5-turbo"):

    # Add user input to message history
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    msg = response.choices[0].message.content

    # Incorporate the model-generated reply into the message history. Crucial for maintaining context in subsequent model calls
    messages.append({"role": "assistant", "content": msg})
    return msg

print("messages:\n")
print_json(messages)
get_completion("有没有土豪套餐？")
print("messages:\n")
print_json(messages)
get_completion("多少钱？")
print("messages:\n")
print_json(messages)
get_completion("给我办一个")
print("messages:\n")
print_json(messages)
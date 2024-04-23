# Example of Function Calling in Simple Math
"""
Calling a local function
Requirement: Implement an AI that can answer questions. If there is addition in the question, it must be able to calculate accurately.
My understanding: The function passed to GPT only has simple function descriptions and parameter definitions, etc. GPT determines which function to choose and returns the parameters based on these.
According to the function selection and parameters returned by GPT, define and execute the corresponding specific function externally, append the result to the message, and let GPT generate the answer again.
"""

"""
Notes on Function Calling:
Only gpt-3.5-turbo-1106 and gpt-4-1106-preview can use the method introduced this time.
gpt-3.5-turbo is an alias for gpt-3.5-turbo-1106.
gpt-4 and gpt-4-1106-preview are two different models.
It is recommended to use the model aliases gpt-3.5-turbo and gpt-4-preview to keep using the latest models.
The description of functions and parameters in Function Calling is also a kind of Prompt.
This kind of Prompt also needs to be optimized, otherwise it will affect the recall of functions, the accuracy of parameters, and even cause GPT to hallucinate.
OpenAI has fine-tuned for Function Calling to ensure the correctness of function call parameters as much as possible.
Function declaration consumes tokens. You need to find the best balance between function coverage, cost saving, and saving context window.
Function Calling can not only call read functions, but also write functions. But the official strongly recommends that someone must confirm before writing.
"""

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

_ = load_dotenv(find_dotenv())

client = OpenAI()


def print_json(data):
    """
    打印参数。如果参数是有结构的（如字典或列表），则以格式化的 JSON 形式打印；
    否则，直接打印该值。
    """
    if hasattr(data, 'model_dump_json'):
        data = json.loads(data.model_dump_json())

    if (isinstance(data, (list, dict))):
        print(json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,  # 模型输出的随机性，0 表示随机性最小
        tools=[{  # 用 JSON 描述函数。可以定义多个。由大模型决定调用谁。也可能都不调用
            "type": "function",
            "function": {
                "name": "sum",
                "description": "加法器，计算一组数的和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            }
                        }
                    }
                }
            }
        }],
    )
    return response.choices[0].message

from math import *

prompt = "Tell me the sum of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
# prompt = "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？"
# prompt = "1+2+3...+99+100"
# prompt = "1024 乘以 1024 是多少？"   # Tools 里没有定义乘法，会怎样？
# prompt = "太阳从哪边升起？"           # 不需要算加法，会怎样？

messages = [
    {"role": "system", "content": "你是一个数学家"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)

# 把大模型的回复加入到对话历史中
print_json(messages)
messages.append(response)

print("=====GPT回复=====")
print_json(response)

# 如果返回的是函数调用结果，则打印出来
if (response.tool_calls is not None):
    # 是否要调用 sum
    tool_call = response.tool_calls[0]
    if (tool_call.function.name == "sum"):
        # 调用 sum
        args = json.loads(tool_call.function.arguments)
        result = sum(args["numbers"])
        print("=====函数返回=====")
        print(result)

        # 把函数调用结果加入到对话历史中
        messages.append(
            {
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": "sum",
                "content": str(result)  # 数值 result 必须转成字符串
            }
        )

        # 再次调用大模型
        print("=====最终回复=====")
        print(get_completion(messages).content)
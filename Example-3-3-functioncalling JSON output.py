# An example of obtaining JSON structure with Function Calling
"""
The stability of generating JSON with Function Calling is relatively high.
For example: Extracting contact name, address, and phone number from a piece of text
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
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        tools=[{
            "type": "function",
            "function": {
                "name": "add_contact",
                "description": "添加联系人",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "联系人姓名"
                        },
                        "address": {
                            "type": "string",
                            "description": "联系人地址"
                        },
                        "tel": {
                            "type": "string",
                            "description": "联系人电话"
                        },
                    }
                }
            }
        }],
    )
    return response.choices[0].message


prompt = "帮我寄给李三，地址是北京市海淀区复兴路55号，电话15233377357。"
messages = [
    {"role": "system", "content": "你是一个通讯录记录入员。"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)
print("====GPT回复====")
print_json(response)
args = json.loads(response.tool_calls[0].function.arguments)
print("====函数参数====")
print_json(args)
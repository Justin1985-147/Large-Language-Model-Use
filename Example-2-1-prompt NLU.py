# This is an example utilizing GPT prompt for Natural Language Understanding (NLU).
#——————————————————————————————————————————————————————————————————
input_text = "有没有便宜的套餐"
# input_text = "有没有土豪套餐"
# input_text = "办个200G的套餐"
# input_text = "有没有流量大的套餐"
# input_text = "200元以下，流量大的套餐有啥"
# input_text = "你说那个10G的套餐，叫啥名字"

# Import required libraries
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables defined in the .env file
_ = load_dotenv(find_dotenv())

# Initialize the OpenAI client
client = OpenAI()  # Uses the OPENAI_API_KEY and OPENAI_BASE_URL from the environment variables by default

# Generate text based on a prompt using the specified model
# Default model is gpt-3.5-turbo
def get_completion(prompt, response_format="text", model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]  # Treat the prompt as user input
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,                                 # Controls the randomness of model output; 0 indicates minimal randomness
        response_format={"type": response_format},
    )
    return response.choices[0].message.content           # Return the generated text from the model

# Task description includes Chinese identifiers for fields
instruction = """
您的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称(name)，月费价格(price)，月流量(data)。
根据用户输入，识别用户在上述三种属性上的倾向。
"""

# Output format includes Chinese definitions and constraints
output_format = """
以JSON格式输出。
1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐 或 null；

2. price字段的取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型

3. data字段的取值为取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型或string类型，string类型只能是'无上限'

4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
(1) 结构体中以"ordering"="descend"表示按降序排序，以"value"字段存储待排序的字段
(2) 结构体中以"ordering"="ascend"表示按升序排序，以"value"字段存储待排序的字段

输出中只包含用户提及的字段，不要猜测任何用户未直接提及的字段，不输出值为null的字段。
"""

examples = """
便宜的套餐：{"sort":{"ordering"="ascend","value"="price"}}
有没有不限流量的：{"data":{"operator":"==","value":"无上限"}}
流量大的：{"sort":{"ordering"="descend","value"="data"}}
100G以上流量的套餐最便宜的是哪个：{"sort":{"ordering"="ascend","value"="price"},"data":{"operator":">=","value":100}}
月费不超过200的：{"price":{"operator":"<=","value":200}}
就要月费180那个套餐：{"price":{"operator":"==","value":180}}
经济套餐：{"name":"经济套餐"}
土豪套餐：{"name":"无限套餐"}
"""

prompt = f"""
{instruction}

{output_format}

用户输入：
{input_text}
"""

# Prompt now includes examples
prompt = f"""
{instruction}

{output_format}

例如：
{examples}

用户输入：
{input_text}
"""

response = get_completion(prompt)
print(response)
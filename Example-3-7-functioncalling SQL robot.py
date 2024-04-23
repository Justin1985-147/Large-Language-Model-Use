# An example of a mobile phone package customer service robot based on Function Calling
"""
Querying the database through Function Calling
Requirement: Find the package that meets the user's needs from the package table.
"""

# prompt = "我是个在校生，有没有套餐推荐？"
# prompt = "流量100G以上，最便宜的是什么套餐？"
prompt = "我不是学生，请问流量100G以上，最便宜的是什么套餐？"

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

def get_sql_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        tools=[{  # 摘自 OpenAI 官方示例 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "Use this function to answer user questions about mobile data plan. \
                            Output should be a fully formed SQL query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            The query should only contain grammars supported by SQLite.
                            """,
                        }
                    },
                    "required": ["query"],
                }
            }
        }],
    )
    return response.choices[0].message

#  描述数据库表结构
database_schema_string = """
CREATE TABLE plan (
    name STR PRIMARY KEY NOT NULL, -- 主键，姓名，不允许为空
    price INT NOT NULL, -- 月费价格，整数类型，不允许为空
    data INT NOT NULL, -- 月流量，整数类型，不允许为空
    requirement STR -- 特殊要求，可以为空    
);
"""

import sqlite3

# 创建数据库连接
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# 创建orders表
cursor.execute(database_schema_string)

# 插入5条明确的模拟记录
mock_data = [
    ("经济套餐", 50, 10, None),
    ("畅游套餐", 180, 100, None),
    ("无限套餐", 300, 1000, None),
    ("校园套餐", 150, 200, "在校生")
]

for record in mock_data:
    cursor.execute('''
    INSERT INTO plan (name, price, data, requirement)
    VALUES (?, ?, ?, ?)
    ''', record)

# 提交事务
conn.commit()

def ask_database(query):
    cursor.execute(query)
    records = cursor.fetchall()
    return records

messages = [
    {"role": "system", "content": "基于plan表回答用户问题，在向用户推荐校园套餐前需要核实用户是否为在校生"},
    {"role": "user", "content": prompt}
]
response = get_sql_completion(messages)
if response.content is None:
    response.content = ""
messages.append(response)
print("====Function Calling====")
print_json(response)

if response.tool_calls is not None:
    tool_call = response.tool_calls[0]
    if tool_call.function.name == "ask_database":
        arguments = tool_call.function.arguments
        args = json.loads(arguments)
        print("====SQL====")
        print(args["query"])
        result = ask_database(args["query"])
        print("====DB Records====")
        print(result)
        
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": "ask_database",
            "content": str(result)
        })

        if any('校园套餐' in item for item in result):
            messages.append({
                "role": "system",
                "content": "基于plan表回答用户问题，在向用户推荐校园套餐前需要核实用户是否为在校生。"
            }
            )
        print(messages)

        response = get_sql_completion(messages)
        print("====最终回复====")
        print(response.content)
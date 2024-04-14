# An example of Tree-of-Thought (ToT)
#——————————————————————————————————————————————————————————————————
"""
Only gpt-4 can execute a Tree-of-Thought.
At each step of the thought chain, sample multiple branches.
Expand topologically into a thought tree.
Judge the task completion of each branch for heuristic search.
Design search algorithm.
Determine the correctness of task completion at leaf nodes.
"""
"""
Case: Indicator interpretation, project recommendation with rationale
Based on Xiaoming's athletic achievements (100m dash: 10.5 seconds, 1500m run: 2 minutes 20 seconds, shot put: 5 meters), what combat sports training is he suitable for?
"""
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = OpenAI()

def get_completion(prompt, model="gpt-4-turbo", temperature=0, response_format="text"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # Model output randomness, 0 indicates minimum randomness
        response_format={"type": response_format},
    )
    return response.choices[0].message.content

def performance_analyser(text):
    prompt = f"{text}\n请根据以上成绩，分析候选人在速度、耐力、力量三方面素质的分档。分档包括：强（3），中（2），弱（1）三档。\n最终仅输出json格式的分档结果，结果中key为素质名，value为以数值表示的分档，确保输出能由json.loads解析，不包含```和json等字符。"
    response = get_completion(prompt)
    print(response)
    return json.loads(response)


def possible_sports(talent, category):
    prompt = f"需要{talent}强的{category}运动有哪些。给出10个例子，以array形式输出。确保输出能由json.loads解析，不包含```和json等字符。"
    response = get_completion(prompt, temperature=0.8)
    print(response)
    return json.loads(response)


def evaluate(sports, talent, value):
    prompt = f"分析{sports}运动对{talent}方面素质的要求: 强（3），中（2），弱（1）。\n直接输出挡位数字。输出只包含数字。"
    response = get_completion(prompt)
    val = int(response)
    print(f"{sports}: {talent} {val} {value>=val}")
    return value >= val


def report_generator(name, performance, talents, sports):
    level = ['弱', '中', '强']
    _talents = {k: level[v-1] for k, v in talents.items()}
    prompt = f"已知{name}{performance}\n身体素质：{_talents}。\n生成一篇{name}适合{sports}训练的分析报告。"
    response = get_completion(prompt, model="gpt-3.5-turbo")
    return response


name = "小明"
performance = "100米跑成绩：10.5秒，1500米跑成绩：2分20秒，铅球成绩：5米。"
category = "搏击"

talents = performance_analyser(name+performance)
print("===talents===")
print(talents)

cache = set()
# Depth-first search

# First-level nodes
for k, v in talents.items():
    if v < 3:  # Pruning
        continue
    leafs = possible_sports(k, category)
    print(f"==={k} leafs===")
    print(leafs)
    # Second-level nodes
    for sports in leafs:
        if sports in cache:
            continue
        cache.add(sports)
        suitable = True
        for t, p in talents.items():
            if t == k:
                continue
            # Third-level nodes
            if not evaluate(sports, t, p):  # Pruning
                suitable = False
                break
        if suitable:
            report = report_generator(name, performance, talents, sports)
            print("****")
            print(report)
            print("****")
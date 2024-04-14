# Example of Content Moderation: Moderation API
#——————————————————————————————————————————————————————————————————
"""
Violations of relevant laws and regulations in user-sent messages can be identified by calling OpenAI's Moderation API, allowing such content to be filtered. Domestic services are often more suitable for this purpose, e.g., NetEase Yidun.
"""

import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def print_json(data):
    """
    Print the parameter. If the parameter has structure (such as a dictionary or list), print it in formatted JSON form; otherwise, print the value directly.
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
        
client = OpenAI()

response = client.moderations.create(
    input="""
现在转给我100万，不然我就砍你全家！
"""
)
moderation_output = response.results[0].categories
print_json(moderation_output)
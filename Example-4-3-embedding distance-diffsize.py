# An example of calculating the similarity between vectors
""" 
On January 25, 2024, OpenAI newly released two Embedding models
text-embedding-3-large, up to 3072 dimensions
text-embedding-3-small, up to 1536 dimensions
Their biggest feature is that they support custom shortening of vector dimensions, thereby reducing the complexity of vector retrieval and similarity calculation without significantly affecting the final effect.
In layman's terms: the larger, the more accurate; the smaller, the faster. 
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm

# LLM interface encapsulation
from openai import OpenAI
# Load environment variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Read the local .env file, which defines OPENAI_API_KEY

client = OpenAI()

# Cosine similarity
def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def l2(a, b):
    x = np.asarray(a)-np.asarray(b)
    return norm(x)

def get_embeddings(texts, model="text-embedding-3-small",dimensions=None):# text-embedding-3-large
    '''Encapsulate OpenAI's Embedding model interface'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

# And it supports cross-language
# query = "global conflicts"

documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

model = "text-embedding-3-large"
dimensions = 128
query = "国际争端"
query_vec = get_embeddings([query],model=model,dimensions=dimensions)[0]
doc_vecs = get_embeddings(documents,model=model,dimensions=dimensions)

print("Dim: {}".format(len(query_vec)))

print("Cosine distance between Query and itself: {:.2f}".format(cos_sim(query_vec, query_vec)))
print("Cosine distance between Query and Documents:")
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))

print()

print("Euclidean distance between Query and itself: {:.2f}".format(l2(query_vec, query_vec)))
print("Euclidean distance between Query and Documents:")
for vec in doc_vecs:
    print(l2(query_vec, vec))
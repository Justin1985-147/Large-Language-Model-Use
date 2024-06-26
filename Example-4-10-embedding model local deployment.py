# Example of local deployment of RAG vector model

# Not every Embedding model is effective for both cosine distance and Euclidean distance
# Which similarity calculation is effective depends on the model's documentation (usually cosine distance calculation is supported)

from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(a, b):
    '''余弦距离 -- 越大越相似'''
    return dot(a, b)/(norm(a)*norm(b))


def l2(a, b):
    '''欧式距离 -- 越小越相似'''
    x = np.asarray(a)-np.asarray(b)
    return norm(x)

model = SentenceTransformer('BAAI/bge-large-zh-v1.5') #中文
# model_name = 'moka-ai/m3e-base'  # 中英双语，但效果一般
# model_name = 'BAAI/bge-m3' # 多语言，但效果一般

query = "国际争端"
# query = "global conflicts"

documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = model.encode(query, normalize_embeddings=True)

doc_vecs = [
    model.encode(doc, normalize_embeddings=True)
    for doc in documents
]

print("Cosine distance:")  # 越大越相似
print(cos_sim(query_vec, query_vec))
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))

print("Euclidean distance:")  # 越小越相似
print(l2(query_vec, query_vec))
for vec in doc_vecs:
    print(l2(query_vec, vec))
# RAG Hybrid Search Example (混合检索的例子)
"""
在实际生产中，传统的关键字检索（稀疏表示）与向量检索（稠密表示）各有优劣。
举个具体例子，比如文档中包含很长的专有名词，关键字检索往往更精准而向量检索容易引入概念混淆。
所以，有时候我们需要结合不同的检索算法，来达到比单一检索算法更优的效果。这就是混合检索。
混合检索的核心是，综合文档d在不同检索算法下的排序名次（rank），为其生成最终排序。
一个最常用的算法叫 Reciprocal Rank Fusion（RRF）
RAG-Fusion 就是利用了 RRF 的原理来提升检索的准确性。
很多向量数据库都支持混合检索，比如 Weaviate、Pinecone 等。也可以根据上述原理自己实现。
"""

query = "非小细胞肺癌的患者"

documents = [
    "李某患有肺癌，癌细胞已转移",
    "刘某肺癌I期",
    "张某经诊断为非小细胞肺癌III期",
    "小细胞肺癌是肺癌的一种"
]

# 1.基于关键字检索的排序

import time  # 导入 time 模块，用于操作时间
from elasticsearch7 import Elasticsearch, helpers

class MyEsConnector:  # 定义一个名为 MyEsConnector 的类
    def __init__(self, es_client, index_name, keyword_fn):  # 初始化方法，接收三个参数：es_client（Elasticsearch 客户端），index_name（索引名称），keyword_fn（关键词函数）
        self.es_client = es_client  # 将 es_client 参数赋值给实例的 es_client 属性
        self.index_name = index_name  # 将 index_name 参数赋值给实例的 index_name 属性
        self.keyword_fn = keyword_fn  # 将 keyword_fn 参数赋值给实例的 keyword_fn 属性
    
    def add_documents(self, documents):  # 定义一个名为 add_documents 的方法，接收一个参数：documents（文档列表）
        '''文档灌库'''  # 方法的注释：文档灌库
        if self.es_client.indices.exists(index=self.index_name):  # 如果索引已经存在
            self.es_client.indices.delete(index=self.index_name)  # 删除索引
        self.es_client.indices.create(index=self.index_name)  # 创建索引
        actions = [  # 定义一个名为 actions 的列表，用于存储批量操作的数据
            {
                "_index": self.index_name,  # 索引名称
                "_source": {  # 文档源数据
                    "keywords": self.keyword_fn(doc),  # 关键词，通过 keyword_fn 函数处理 doc 得到
                    "text": doc,  # 文本，直接使用 doc
                    "id": f"doc_{i}"  # 文档 ID，使用字符串格式化生成
                }
            }
            for i,doc in enumerate(documents)  # 遍历 documents，同时获取元素的索引和值
        ]
        helpers.bulk(self.es_client, actions)  # 使用 helpers.bulk 方法批量添加文档
        time.sleep(1)  # 暂停 1 秒，等待 Elasticsearch 处理完批量操作

    def search(self, query_string, top_n=3):  # 定义一个名为 search 的方法，接收两个参数：query_string（查询字符串），top_n（返回的文档数量，默认为 3）
        '''检索'''  # 方法的注释：检索
        search_query = {  # 定义一个名为 search_query 的字典，用于存储查询条件
            "match": {  # 匹配查询
                "keywords": self.keyword_fn(query_string)  # 关键词，通过 keyword_fn 函数处理 query_string 得到
            }
        }
        res = self.es_client.search(index=self.index_name, query=search_query, size=top_n)  # 使用 Elasticsearch 客户端的 search 方法进行搜索，返回的结果存储在 res 中
        return { 
            hit["_source"]["id"] : {  # 以文档的 ID 为键
                "text" : hit["_source"]["text"],  # 文本
                "rank" : i,  # 排名
            }
            for i, hit in enumerate(res["hits"]["hits"])  # 遍历搜索结果，同时获取元素的索引和值
        }
    
from chinese_and_english_utils import to_keywords # 使用中文、英文的关键字提取函数

import os

# Introduce configuration file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Read the local .env file, which defines ELASTICSEARCH_BASE_URL, ELASTICSEARCH_PASSWORD, ELASTICSEARCH_NAME, OPENAI_API_KEY
ELASTICSEARCH_BASE_URL = os.getenv('ELASTICSEARCH_BASE_URL')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')
ELASTICSEARCH_NAME= os.getenv('ELASTICSEARCH_NAME')
print(ELASTICSEARCH_BASE_URL)
# print(ELASTICSEARCH_PASSWORD)
# print(ELASTICSEARCH_NAME)

# 1. Create Elasticsearch connection
es = Elasticsearch(
    hosts=[ELASTICSEARCH_BASE_URL],  # Service address and port
    http_auth=(ELASTICSEARCH_NAME, ELASTICSEARCH_PASSWORD),  # Username, password
)

# 创建 ES 连接器
es_connector = MyEsConnector(es, "demo_es_lq", to_keywords)

# 文档灌库
es_connector.add_documents(documents)

# 关键字检索
keyword_search_results = es_connector.search(query, 3)

print(keyword_search_results)

# 2.基于向量检索的排序
import chromadb
from chromadb.config import Settings

class MyVectorDBConnector: #内存模式
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results

from openai import OpenAI

client = OpenAI()

def get_embeddings(texts, model="text-embedding-3-small",dimensions=None):# text-embedding-3-large
    '''封装 OpenAI 的 Embedding 模型接口'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

# 创建向量数据库连接器
vecdb_connector = MyVectorDBConnector("demo_vec_lq", get_embeddings)

# 文档灌库
vecdb_connector.add_documents(documents)

# 向量检索
vector_search_results = {
    "doc_"+str(documents.index(doc)) : {
        "text" : doc,
        "rank" : i
    }
    for i, doc in enumerate(
        vecdb_connector.search(query, 3)["documents"][0]
    )
} # 把结果转成跟上面关键字检索结果一样的格式

print(vector_search_results)

# 3.基于 RRF 的融合排序
def rrf(ranks, k=1):
    ret = {}
    # 遍历每次的排序结果
    for rank in ranks: 
        # 遍历排序中每个元素
        for id, val in rank.items():
            if id not in ret:
                ret[id] = { "score": 0, "text": val["text"] }
            # 计算 RRF 得分
            ret[id]["score"] += 1.0/(k+val["rank"])
    # 按 RRF 得分排序，并返回
    return dict(sorted(ret.items(), key=lambda item: item[1]["score"], reverse=True))

import json

# 融合两次检索的排序结果
reranked = rrf([keyword_search_results,vector_search_results])

print(json.dumps(reranked,indent=4,ensure_ascii=False))
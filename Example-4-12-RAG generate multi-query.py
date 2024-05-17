# An example of RAG (Retrieval-Augmented Generation) improving retrieval efficiency by generating multiple queries

# RAG based on local chroma vector library (non-memory mode)
# Here we use the embedding and dialogue interface of openai
# First, you need to start the chroma vector database server in the terminal, and then you can access it, avoiding the trouble of having to rebuild the vector database every time
# chroma run --path D:\VectorDataBase

# !pip install sentence_transformers

# user_query = "how safe is llama 2"
user_query = "llama 2可以商用吗？"
# user_query = "llama 2有对话版吗？"
# user_query = "how many parameters does llama 2 have?"
isFirstRun = False # Whether it is the first run, if it is, you need to build a vector database
isResultSort = True # Whether to generate multiple queries and sort the results 
chunk_size=300 # Length of the text to be cut
overlap_size=100 # Length of the overlap
top_n=2 # The number of results selected after sorting
top_nc=5 # The number of results returned for each query
n_queries=4 # The number of multiple queries generated based on the original query


from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_text_from_pdf(filename, page_numbers=None, min_line_length=10):
    '''Extract text from a PDF file (by specified page number)'''
    paragraphs = []
    buffer = ''
    full_text = ''
    # Extract all text
    for i, page_layout in enumerate(extract_pages(filename)):
        # If a page range is specified, skip pages outside the range
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # Separate by blank lines and reorganize the text into paragraphs
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs

import chromadb

class MyVectorDBConnector: # Memory mode
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.HttpClient(host='localhost', port=8000)
           
        # No need to clear previous content
        # chroma_client.reset()

        # Create a collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"}) #hnsw:space can be cosine or l2, the default is l2
        self.embedding_fn = embedding_fn

    def add_documents(self, documents, metadatainputs=None):
        '''Add documents and vectors to the collection'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # Vector of each document
            documents=documents,  # Original text of the document
            metadatas=[{"source": metadatainputs} for _ in documents],  # Metadata of each document
            ids=[f"id{i}" for i in range(len(documents))]  # id of each document
        )

    def search(self, query, top_n):
        '''Search the vector database'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results

from openai import OpenAI
# Load environment variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Read the local .env file, which defines the OPENAI_API_KEY

client = OpenAI()

# Function to generate queries using OpenAI's ChatGPT
def generate_queries_chatgpt(original_query, model="gpt-3.5-turbo", n_queries=4):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": f"OUTPUT ({n_queries} queries):"}
        ]
    )
    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries
def get_embeddings(texts, model="text-embedding-3-small",dimensions=None):# text-embedding-3-large
    '''Encapsulate the Embedding model interface of OpenAI'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

def get_completion(prompt, model="gpt-3.5-turbo"):
    '''Encapsulate the openai interface'''
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # The randomness of the model output, 0 means the least randomness
    )
    return response.choices[0].message.content

def build_prompt(prompt_template, **kwargs):
    '''Assign values to the Prompt template'''
    # Initialize an empty dictionary to store the processed input
    inputs = {}

    # Traverse the input keyword parameters
    for k, v in kwargs.items():
        # If the value of the parameter is a list of strings
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            # Join all the strings in the list with two newline characters
            val = '\n\n'.join(v)
        else:
            # Otherwise, use the value of the parameter directly
            val = v
        # Add the processed value to the dictionary
        inputs[k] = val

    # Use the processed input parameters to format the template, and return the result
    return prompt_template.format(**inputs)

# Prompt template
prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{info}

用户问：
{query}

请用中文回答用户问题。
"""

from nltk.tokenize import sent_tokenize  # Import the sent_tokenize function from the NLTK library for splitting text into sentences

def split_text(paragraphs, chunk_size=300, overlap_size=100):  # Define a function that takes a list of paragraphs and two optional parameters: chunk_size and overlap_size
    '''Split the text by the specified chunk_size and overlap_size'''  # The function's docstring, explaining what the function does
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]  # Split each paragraph into sentences, and remove leading and trailing whitespace from each sentence
    chunks = []  # Initialize an empty list to store the generated chunks
    i = 0  # Initialize a variable i for indexing in the list of sentences
    while i < len(sentences):  # Execute the loop while i is less than the length of the list of sentences
        chunk = sentences[i]  # Take the i-th sentence from the list of sentences as the current chunk
        overlap = ''  # Initialize an empty string to store the overlap with the previous chunk
        prev = i - 1  # Initialize a variable prev, representing the index of the sentence before the current one
        # Calculate the overlap forward
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:  # Execute the loop while prev is not less than 0 and the length of the previous sentence plus the length of the overlap does not exceed overlap_size
            overlap = sentences[prev] + ' ' + overlap  # Add the previous sentence to the front of the overlap
            prev -= 1  # Decrement prev by 1, moving to the next sentence forward
        chunk = overlap+chunk  # Add the overlap to the front of the current chunk
        next = i + 1  # Initialize a variable next, representing the index of the sentence after the current one
        # Calculate the current chunk backward
        while next < len(sentences) and len(sentences[next])+len(chunk) <= chunk_size:  # Execute the loop while next is less than the length of the list of sentences and the length of the next sentence plus the length of the current chunk does not exceed chunk_size
            chunk = chunk + ' ' + sentences[next]  # Add the next sentence to the end of the current chunk
            next += 1  # Increment next by 1, moving to the next sentence backward
        chunks.append(chunk)  # Add the current chunk to the list of chunks
        i = next  # Set i to next, i.e., move to the next sentence to be processed
    return chunks  # Return the generated list of chunks

class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. Search
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. Build Prompt
        prompt = build_prompt(
            prompt_template, info=search_results['documents'][0], query=user_query)
        print("====Prompt====\n")
        print(prompt)

        # 3. Call LLM
        response = self.llm_api(prompt)
        return response
    
class RAG_BotC:
    def __init__(self, vector_db, llm_api, n_results=5):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query, search_results):
        # 1. Build Prompt
        prompt = build_prompt(
            prompt_template, info=search_results, query=user_query)
        print("====Prompt====\n")
        print(prompt)

        # 2. Call LLM
        response = self.llm_api(prompt)
        return response

# Create or associate a vector database object
vector_db = MyVectorDBConnector("demo_split", get_embeddings)

if isFirstRun:
    # Extract text from PDF
    paragraphs = extract_text_from_pdf("llama2.pdf", page_numbers=[
                                      2, 3], min_line_length=10)
    chunks = split_text(paragraphs, 300, 100)
    # Add documents to the vector database
    vector_db.add_documents(chunks, metadatainputs="llama2.pdf")

if isResultSort==False:
    # Create a RAG bot
    bot = RAG_Bot(
        vector_db,
        llm_api=get_completion,
        n_results=top_n
    )
    search_results = vector_db.search(user_query, top_n)
    for doc in search_results['documents'][0]:
        print(doc+"\n")
    print("====Reply====")
    print(bot.chat(user_query)) 
else:
    from sentence_transformers import CrossEncoder
    generated_queries = generate_queries_chatgpt(user_query, n_queries=n_queries)
    generated_queries.insert(0, 'original. "' + user_query + '"')
    print("====原始查询及生成的查询====\n")
    print(generated_queries)
    
    search_results = []
    # Vector search
    for query in generated_queries:
        print("====查询====\n")
        print(query)
        tresult=vector_db.search(query, top_nc)
        if tresult is None:
            continue
        print(tresult["ids"][0])
        vector_search_results = {
            "doc_"+doc_id: {
                "text" : doc,
                "rank" : i
            }          
            for i, (doc, doc_id) in enumerate(
                zip(tresult["documents"][0],tresult["ids"][0])
            )
        }
        print(vector_search_results)
        search_results.append(vector_search_results)
    #print("====向量检索结果====\n")
    #print(search_results)

    # 基于 RRF 的融合排序
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
    reranked = rrf(search_results)
    print("====融合排序结果====\n")
    print(json.dumps(reranked,indent=4,ensure_ascii=False))
    
    # Create a RAG bot
    bot = RAG_BotC(
        vector_db,
        llm_api=get_completion,
        n_results=top_nc
    )
    print("====回复====")
    reranked_list = list(reranked.items())
    print(bot.chat(user_query,[text for score, text in reranked_list[:top_n]]))
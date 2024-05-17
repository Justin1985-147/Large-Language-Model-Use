# RAG example based on vector search
# Here we use Wenxin Qianfan's embedding and dialogue interface

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
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
    # Reorganize the text into paragraphs separated by blank lines
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
from chromadb.config import Settings

class MyVectorDBConnector: # Memory mode
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # For demonstration, no need to reset() every time in reality
        chroma_client.reset()

        # Create a collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''Add documents and vectors to the collection'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # Vector of each document
            documents=documents,  # Original text of the document
            ids=[f"id{i}" for i in range(len(documents))]  # id of each document
        )

    def search(self, query, top_n):
        '''Search the vector database'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results

import json
import requests
import os

from dotenv import load_dotenv, find_dotenv
# Load environment variables defined in .env file
_ = load_dotenv(find_dotenv())
# Get access token through authentication interface
# Need to register with Baidu to get two keys, then set the keys as the corresponding environment variables to ensure they are not directly exposed
def get_access_token():
    """
    Generate authentication signature (Access Token) using AK, SK
    :return: access_token, or None(if error)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": os.getenv('ERNIE_CLIENT_ID'),
        "client_secret": os.getenv('ERNIE_CLIENT_SECRET')
    }
    return str(requests.post(url, params=params).json().get("access_token"))

# Call Wenxin Qianfan BGE Embedding interface
def get_embeddings_bge(prompts):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/bge_large_en?access_token=" + get_access_token()
    payload = json.dumps({
        "input": prompts
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request(
        "POST", url, headers=headers, data=payload).json()
    data = response["data"]
    return [x["embedding"] for x in data]

# Call Wenxin 4.0 dialogue interface
def get_completion_ernie(prompt):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {'Content-Type': 'application/json'}
    response = requests.request(
        "POST", url, headers=headers, data=payload).json()
    return response["result"]

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
You are a question answering robot.
Your task is to answer user questions based on the given information below.
Make sure your reply is entirely based on the information below. Do not make up answers.
If the information below is not enough to answer the user's question, please reply directly "I can't answer your question".

Known information:
{info}

User asks:
{query}

Please answer the user's question in English.
"""

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

        # 3. Call LLM
        response = self.llm_api(prompt)
        return response
  
# For demonstration convenience, we only take two pages (Chapter 1)
paragraphs = extract_text_from_pdf("llama2.pdf", page_numbers=[
                                   2, 3], min_line_length=10)

# Create a vector database object
new_vector_db = MyVectorDBConnector(
    "demo_ernie",
    embedding_fn=get_embeddings_bge
)
# Add documents to the vector database
new_vector_db.add_documents(paragraphs)

# Create a RAG robot
new_bot = RAG_Bot(
    new_vector_db,
    llm_api=get_completion_ernie
)

user_query = "how many parameters does llama 2 have?"
response = new_bot.chat(user_query)
print(response)
# RAG example based on vector retrieval
# Here we use openai's embedding and dialogue interface
# Vector database
# Vector database, is middleware specifically designed for vector retrieval
# Clarify a few key concepts:
# The significance of the vector database is fast retrieval;
# The vector database itself does not generate vectors, vectors are produced by the Embedding model;
# The vector database is complementary to the traditional relational database, not a substitute, often used together in actual applications according to actual needs.

"""
RAG process
Offline steps:
Document loading
Document segmentation
Vectorization
Pour into vector database

Online steps:
Get user questions
Vectorize user questions
Retrieve vector database
Fill the retrieved results and user questions into the Prompt template
Call LLM with the final obtained Prompt
Generate reply by LLM 
"""

""" 
What if the open source RAG I used doesn't work?
Check preprocessing results: whether the document is loaded correctly, whether the segmentation is reasonable
Test retrieval results: whether the text fragments retrieved by the question contain the answer
Test the ability of the large model: given the premise of the question and the text fragment containing the answer, can the large model correctly answer the question 
"""

# !pip install chromadb

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''Extract text from PDF file (by specified page number)'''
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
        '''Retrieve vector database'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results

from openai import OpenAI
# Load environment variables
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Read the local .env file, which defines OPENAI_API_KEY

client = OpenAI()

def get_embeddings(texts, model="text-embedding-3-small",dimensions=None):# text-embedding-3-large
    '''Encapsulate OpenAI's Embedding model interface'''
    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

def get_completion(prompt, model="gpt-3.5-turbo"):
    '''Encapsulate openai interface'''
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # The randomness of the model output, 0 means the smallest randomness
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
        # 1. Retrieval
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
vector_db = MyVectorDBConnector("demo", get_embeddings)
# Add documents to the vector database
vector_db.add_documents(paragraphs)

# Create a RAG robot
bot = RAG_Bot(
    vector_db,
    llm_api=get_completion
)

user_query = "Does llama 2 have a dialogue version?"
response = bot.chat(user_query)
print(response)
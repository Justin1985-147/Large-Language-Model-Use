# RAG: An example of a retrieval-based question answering model based on ES
# Limitations of keyword search: The same semantics, different wording, may lead to ineffective results

# Install ES client
# !pip install elasticsearch7
# Install NLTK (text processing method library)
# !pip install nltk

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from elasticsearch7 import Elasticsearch, helpers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

import warnings
warnings.simplefilter("ignore")  # Shield some warnings of ES

# VPN required, no need to download repeatedly after downloading
# nltk.download('punkt')  # English word segmentation, root, sentence segmentation, etc.
# nltk.download('stopwords')  # English stop word library

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
    # Separate the text into paragraphs by blank lines
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

# Here to_keywords is implemented for English, for Chinese implementation please refer to chinese_utils.py
def to_keywords(input_string):
    '''(English) Text only retains keywords'''
    # Use regular expressions to replace all non-alphanumeric characters with spaces
    no_symbols = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string)
    word_tokens = word_tokenize(no_symbols)
    # Load stop word list
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    # Remove stop words, take roots
    filtered_sentence = [ps.stem(w)
                         for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def search(query_string, top_n=3):
    # ES query language
    search_query = {
        "query": {
            "match": {
                "keywords": to_keywords(query_string)
            }
        }
    }
    res = es.search(index=index_name, body=search_query, size=top_n)
    return [hit["_source"]["text"] for hit in res["hits"]["hits"]]

# Call the extract_text_from_pdf function to extract text from the "llama2.pdf" file,
# and set the minimum line length to 10. The return value of the function (a list containing all paragraphs) is assigned to the variable paragraphs
paragraphs = extract_text_from_pdf("llama2.pdf", min_line_length=10)

# Traverse the first three elements of the paragraphs list (i.e., the first three paragraphs)
for para in paragraphs[:3]:
    # Print each paragraph, and add a newline character after each paragraph
    print(para+"\n")

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

# 2. Define index name
index_name = "teacher_demo_index_lq"

# 3. If the index already exists, delete it (only for demonstration, not needed in actual application)
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# 4. Create index
es.indices.create(index=index_name)

# 5. Library filling instruction
actions = [
    {
        "_index": index_name,
        "_source": {
            "keywords": to_keywords(para),
            "text": para
        }
    }
    for para in paragraphs
]

# 6. Text library filling
success, _ = helpers.bulk(es, actions)
# Print the number of successfully imported documents
print("Successfully indexed %d documents" % success)

# Check if the index exists
index_exists = es.indices.exists(index=index_name)
print("Index exists:", index_exists)

# Print the number of documents in the index
doc_count = es.count(index=index_name)['count']
print("Document count:", doc_count)

# LLM interface encapsulation
from openai import OpenAI
import os

client = OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    '''Encapsulate openai interface'''
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

# Initial exploration of RAG Pipeline
#user_query = "how many parameters does llama 2 have?"
user_query="Does llama 2 have a chat version?"
#user_query = "Does llama 2 have a conversational variant?"


# 1. Retrieval
search_results = search(user_query, 2)

# 2. Build Prompt
prompt = build_prompt(prompt_template, info=search_results, query=user_query)
print("===Prompt===")
print(prompt)

# 3. Call LLM
response = get_completion(prompt)
print("===Response===")
print(response)
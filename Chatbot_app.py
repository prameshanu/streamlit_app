import streamlit as st
## Data Ingestion 
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Replace with appropriate embedding
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.retrievers import PineconeHybridSearchRetriever
import sentence_transformers
from langchain.chains import RetrievalQA

from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone_text.sparse import BM25Encoder


import warnings


import streamlit as st

import numpy as np
from dotenv import load_dotenv
import os

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find
## Data Ingestion 

# Download required resources for NLTK
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

## lazy loading
try:
    # Check if 'punkt' is available; download if not
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    # Check if 'stopwords' is available; download if not
    find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

try:
    # Check if 'wordnet' is available; download if not
    find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

import os

langchain_api_key  = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key


## Preprocessing for original files was done separately and saved separately to reduce processing every time the code is being executed
@st.cache_data
def load_documents_from_github(repo_url):
    from urllib.parse import urljoin

    # Convert GitHub URL to raw content URL
    raw_url_base = repo_url.replace("github.com", "raw.githubusercontent.com").replace("/tree/", "/")

    # List of files to process (You might need to dynamically fetch this)
    file_path = [
        "processed_1.txt",	"processed_2.txt",	"processed_3.txt",	"processed_4.txt",	"processed_5.txt",	"processed_6.txt",	
        "processed_7.txt",	"processed_8.txt",	"processed_9.txt",	"processed_10.txt",	"processed_11.txt",	"processed_12.txt",	
        "processed_13.txt",	"processed_14.txt",	"processed_15.txt",	"processed_16.txt",	"processed_17.txt",	"processed_18.txt",	
        "processed_19.txt",	"processed_20.txt",	"processed_21.txt",	"processed_22.txt",	"processed_23.txt",	"processed_24.txt",	
        "processed_25.txt",	"processed_26.txt",	"processed_27.txt",	"processed_28.txt",	"processed_29.txt",	"processed_30.txt",	
        "processed_31.txt",	"processed_32.txt",	"processed_33.txt",	"processed_34.txt",	"processed_35.txt",	"processed_36.txt",	
        "processed_37.txt",	"processed_38.txt",	"processed_39.txt",	"processed_40.txt",	"processed_41.txt",	"processed_42.txt",	
        "processed_43.txt",	"processed_44.txt",	"processed_45.txt",	"processed_46.txt",	"processed_47.txt",	"processed_48.txt",	
        "processed_49.txt",	"processed_50.txt",	"processed_51.txt",	"processed_52.txt",	"processed_53.txt",	"processed_54.txt",	
        "processed_55.txt",	"processed_56.txt",	"processed_57.txt",	"processed_58.txt",	"processed_59.txt",	"processed_60.txt"
    ]


    docs = []

    def load_file_from_github(file_path):
        raw_file_url = urljoin(raw_url_base, file_path)
        response = requests.get(raw_file_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text

    with ThreadPoolExecutor() as executor:
        for result in executor.map(load_file_from_github, file_paths):
            loader = TextLoader.from_string(result)
            docs.extend(loader.load())

    return docs

# GitHub repo URL containing your files
repo_url = "https://github.com/prameshanu/streamlit_app/tree/main/processed_data"
docs = load_documents_from_github(repo_url)
a = docs[:1]



def preprocess_text(text):
    # Step 1: Lowercase the text
    text = text.lower()
    
    # Step 2: Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    # Step 3: Tokenize the text
    tokens = word_tokenize(text)
    
    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Step 5: Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string (optional)
    processed_text = ' '.join(tokens)
    
    return processed_text



# @st.cache_data
# def load_documents(directory_path):
#     from concurrent.futures import ThreadPoolExecutor

#     def load_file(file_path):
#         loader = TextLoader(file_path)
#         return loader.load()

#     with ThreadPoolExecutor() as executor:
#         file_paths = [
#             os.path.join(directory_path, filename)
#             for filename in os.listdir(directory_path)
#             if os.path.isfile(os.path.join(directory_path, filename))
#         ]
#         docs = []
#         for result in executor.map(load_file, file_paths):
#             docs.extend(result)
#     return docs
    

# directory_path = 'https://raw.githubusercontent.com/prameshanu/streamlit_app/tree/main/processed_data'
# docs = load_documents(directory_path)


# Streamlit Framework
st.title('Langchain Demo incorporating Hybrid Search With LLAMA2 API')

st.write(a)
# # State Initialization
if "done" not in st.session_state:
    st.session_state.done = False  # To track if the user clicked "I am done, Thanks."
if "history" not in st.session_state:
    st.session_state.history = []  # To store the chat history
if "current_question" not in st.session_state:
    st.session_state.current_question = ""  # To store the current input text


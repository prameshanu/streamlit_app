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
from concurrent.futures import ThreadPoolExecutor
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic

from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone_text.sparse.bm25_encoder import BM25Encoder


import warnings
import streamlit as st

import numpy as np
from dotenv import load_dotenv

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find
import requests
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

try:
    # Check if 'punkt_tab' is available; download if not
    find('corpora/punkt_tab.zip')
except LookupError:
    nltk.download('punkt_tab')



langchain_api_key  = st.secrets["LANGCHAIN_API_KEY"]
pine_cone_api_key = st.secrets["PINE_CONE_API_KEY"]
# claude_api_key = st.secrets["CLAUDE_API_KEY"]



### Preprocessing function for input text, for input data: preprocessing was done separately to avoid repeat code execution on every run.

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


### Input 60 processed files

# Base URL for the files
raw_url_base = "https://raw.githubusercontent.com/prameshanu/streamlit_app/main/processed_data/"

# List of file names
file_path = [
    "processed_1.txt", "processed_2.txt", "processed_3.txt", "processed_4.txt", "processed_5.txt", "processed_6.txt", 
    "processed_7.txt", "processed_8.txt", "processed_9.txt", "processed_10.txt", "processed_11.txt", "processed_12.txt", 
    "processed_13.txt", "processed_14.txt", "processed_15.txt", "processed_16.txt", "processed_17.txt", "processed_18.txt", 
    "processed_19.txt", "processed_20.txt", "processed_21.txt", "processed_22.txt", "processed_23.txt", "processed_24.txt", 
    "processed_25.txt", "processed_26.txt", "processed_27.txt", "processed_28.txt", "processed_29.txt", "processed_30.txt", 
    "processed_31.txt", "processed_32.txt", "processed_33.txt", "processed_34.txt", "processed_35.txt", "processed_36.txt", 
    "processed_37.txt", "processed_38.txt", "processed_39.txt", "processed_40.txt", "processed_41.txt", "processed_42.txt", 
    "processed_43.txt", "processed_44.txt", "processed_45.txt", "processed_46.txt", "processed_47.txt", "processed_48.txt", 
    "processed_49.txt", "processed_50.txt", "processed_51.txt", "processed_52.txt", "processed_53.txt", "processed_54.txt", 
    "processed_55.txt", "processed_56.txt", "processed_57.txt", "processed_58.txt", "processed_59.txt", "processed_60.txt"
]

# Directory to temporarily store the downloaded files
download_dir = "./temp_downloads"
os.makedirs(download_dir, exist_ok=True)

# Create an empty list to store documents
documents = []

# Loop through each file, download it, and load using TextLoader
for file in file_path:
    file_url = raw_url_base + file  # Construct the full URL
    local_path = os.path.join(download_dir, file)  # Local path for the downloaded file
    
    # Download the file
    response = requests.get(file_url)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    
    # Use TextLoader to load the file
    loader = TextLoader(local_path)
    docs = loader.load()
    documents.extend(docs)  # Add to the list of documents
    
    # Optionally, remove the file after processing
    # os.remove(local_path)



def preprocess_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=10)
    return text_splitter.split_documents(docs)

documents = preprocess_documents(docs)


# Streamlit Framework
st.title('Langchain Demo incorporating Hybrid Search With LLAMA2 API')


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

langchain_api_key = os.environ("LANGCHAIN_API_KEY")


# Streamlit Framework
st.title('Langchain Demo incorporating Hybrid Search With LLAMA2 API')

# st.write(langchain_api_key)
# # State Initialization
if "done" not in st.session_state:
    st.session_state.done = False  # To track if the user clicked "I am done, Thanks."
if "history" not in st.session_state:
    st.session_state.history = []  # To store the chat history
if "current_question" not in st.session_state:
    st.session_state.current_question = ""  # To store the current input text


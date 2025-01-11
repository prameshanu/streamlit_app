import streamlit as st
## Data Ingestion 
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings  # Replace with appropriate embedding
# from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.retrievers import PineconeHybridSearchRetriever
# import sentence_transformers
# from langchain.chains import RetrievalQA

# from pinecone import Pinecone
# from pinecone import ServerlessSpec
# from pinecone_text.sparse import BM25Encoder



# Streamlit Framework
st.title('Langchain Demo incorporating Hybrid Search With LLAMA2 API')

# # State Initialization
if "done" not in st.session_state:
    st.session_state.done = False  # To track if the user clicked "I am done, Thanks."
if "history" not in st.session_state:
    st.session_state.history = []  # To store the chat history
if "current_question" not in st.session_state:
    st.session_state.current_question = ""  # To store the current input text


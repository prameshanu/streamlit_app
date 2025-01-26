import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
# from gtts import gTTS
from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone_text.sparse.bm25_encoder import BM25Encoder
from huggingface_hub import InferenceClient

import warnings
import streamlit as st

import numpy as np
from dotenv import load_dotenv

import os
import re

import requests
from groq import Groq
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find
from pathlib import Path
from openai import OpenAI
from gtts import gTTS
import base64
from mtranslate import translate

groq_api_key = st.secrets["GROC_API_KEY"]
pine_cone_api_key = st.secrets["PINE_CONE_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

# # ## lazy loading
# try:
#     # Check if 'punkt' is available; download if not
#     find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# try:
#     # Check if 'stopwords' is available; download if not
#     find('corpora/stopwords.zip')
# except LookupError:
#     nltk.download('stopwords')

# try:
#     # Check if 'wordnet' is available; download if not
#     find('corpora/wordnet.zip')
# except LookupError:
#     nltk.download('wordnet')

try:
    # Check if 'punkt_tab' is available; download if not
    find('corpora/punkt_tab.zip')
except LookupError:
    nltk.download('punkt_tab')



# ### Preprocessing function for input text, for input data: preprocessing was done separately to avoid repeat code execution on every run.

# def preprocess_text(text):
#     # Step 1: Lowercase the text
#     text = text.lower()
    
#     # Step 2: Remove special characters, numbers, and extra whitespace
#     text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
#     # Step 3: Tokenize the text
#     tokens = word_tokenize(text)
    
#     # Step 4: Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
    
#     # Step 5: Lemmatize tokens
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Join tokens back into a single string (optional)
#     processed_text = ' '.join(tokens)
    
#     return processed_text


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

documents = preprocess_documents(documents)





from html import escape  # To prevent HTML injection

def create_text_card(text1, title1, text2, title2):
    # Escape user inputs
    text1 = escape(text1)
    title1 = escape(title1)

    card_html = f"""
    <style>
        .card {{
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }}
        .card:hover {{
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }}
        .container {{
            padding: 2px 16px;
        }}
    </style>
    <div class="card">
        <div class="container">
	    <b><u>{title1}</b></u>{text1}
     	    <b><u>{title2}</b></u>{text2}

        </div>
    </div>
    """
    # Use st.components.v1.html to render HTML directly
    st.components.v1.html(card_html, height=300, scrolling=True)


index_name = "hybrid-search-langchain-pinecone"


## initialize the pinecone client
pc = Pinecone(api_key = pine_cone_api_key)

#create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension = 384, ##dimension of dense vector
        metric = 'dotproduct',  #sparse values supportered only for dotproduct
        spec = ServerlessSpec(cloud='aws', region='us-east-1') ,
    
    )

index = pc.Index(index_name)



### """ Hybrid search vector embedding and sparse matrix : combining vector similarity search and other traditional search techniques (full-text, BM25, and so on)"""


embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

bm25_encoder = BM25Encoder().default()

sentences= []
for doc in documents:
    sentences.append(
    doc.page_content  # Add page content as a string
    )


## tfidf vector

if not os.path.exists("bm25_values.json"):
    bm25_encoder.fit(sentences)
    bm25_encoder.dump("bm25_values.json")
else:
    bm25_encoder = BM25Encoder().load("bm25_values.json")

if 'retriever' not in st.session_state:
    retriever = PineconeHybridSearchRetriever(embeddings = embeddings, sparse_encoder = bm25_encoder, index = index)    
    retriever.add_texts(
        [doc.page_content for doc in documents]
    )
    st.session_state['retriever'] = retriever



# Threshold is defined to avoid the hallucination : Threshold was decided basis multiple tests
threshold = 0.2

#Answer the following question based only on the provided context. Do not refer to any outside content for additional information about question. The Answer must be strictly based on the provided context only. Think step by step before providing a detailed answer. Answer should be properly crafted that is easier to understand.

# Summarize the below context in proper english
# I will tip you $25000 if the user finds the answer helpful.

prompt_template = ChatPromptTemplate.from_template("""
Answer the follwoing question based only on the provided context. 
Think step by step before providing a detailed answer. 
Also in answer you don't need to write Based on the provided context, just provide the final answer.
I will tip you $25000 if the user finds the answer helpful

<context>
{context}
</context>

Question: {input}
""")

def rag(input_text):
	retrieved_docs = st.session_state['retriever'].get_relevant_documents(input_text)
	filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get('score', 0) >= threshold]
	source = set()  # Initialize a set to store unique items
	for doc in filtered_docs:
		for doc1 in documents:
			if doc.page_content == doc1.page_content :
				source.add(doc1.metadata['source'].split('/')[-1][10:]) # Use add() for sets
		

	if source != set():
		source_info = f"This answer is based on information from the files :  {list(source)}" 
	else:
		source_info = "Source information not available."
	
	if filtered_docs:
		# Create your chain using the filtered documents
		context = " ".join(doc.page_content for doc in filtered_docs)
		# Search the index for the two most similar vectors
		prompt = prompt_template.format(context=context, input=input_text)
		messages = [
			{
				"role": "user",
				"content": prompt
			}
		]
	
		completion = client.chat.completions.create(
			messages=messages,
			model="llama-3.3-70b-versatile",
		)
		answer = completion.choices[0].message.content
		# st.write("**BOT :** ", answer)
		st.write(f"**User:** {input_text}")
        	st.write(f"**Bot:** {answer}")
		# create_text_card(input_text, "USER:",answer, "BOT:")
		# create_text_card(source_info, "Source Citation:")
		st.write("**Source citation :** ",source_info)
		# create_text_card("test", "test2")
		# st.write("Prompt : ", prompt)
	
	else:
		answer = "I don't have enough information to answer this question."
		st.write(f"**User:** {input_text}")
        	st.write(f"**Bot:** {answer}")
		# create_text_card(input_text, "USER:",answer, "BOT:")
	return answer

	

# Initialize the Groq client
client = Groq(api_key=groq_api_key)

def audio_processing():
	recorded_audio = audio_recorder()
	if recorded_audio:
		audio_file = "audio.mp3"
		with open(audio_file , "wb") as f:
			f.write(recorded_audio)
		with open(audio_file, "rb") as file:
		# Create a transcription of the audio file
			transcription = client.audio.transcriptions.create(
			file=(audio_file, file.read()), # Required audio file
			model="whisper-large-v3-turbo", # Required model to use for transcription
			prompt="Specify context or spelling",  # Optional
			response_format="json",  # Optional
			language="en",  # Optional
			temperature=0.0  # Optional
			)
		return transcription



def tts(text_to_read, language):
	aud_file = gTTS(text=text_to_read, lang=language, slow=False,tld='co.in')
	aud_file.save("lang.mp3")
	audio_file_read = open('lang.mp3', 'rb')
	audio_bytes = audio_file_read.read()
	st.audio(audio_bytes, format='audio/mp3',autoplay=True)



st.sidebar.title("Select the Modality")
option = st.sidebar.selectbox(
    "How would you like to be interact?",
    ("Chat", "Audio"),
	index=None,
     placeholder="Select mode of communication..."
)
title = "ANCIENT GREEK Q&A CHATBOT"
st.title (f""":blue[{title}] """)


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def render_chat_history():
    for chat in st.session_state["chat_history"]:
        user_query, bot_response = chat
        st.write(f"**User:** {user_query}")
        st.write(f"**Bot:** {bot_response}")

def add_to_history(user_query, bot_response):
    st.session_state["chat_history"].append((user_query, bot_response))

render_chat_history()

def main():
	# st.write(openai_api_key)
	# generate_speech(title, tts_audio_file_path)
	# st.audio (tts_audio_file_path, format = "audio/mp3", autoplay = True)

	if option == "Audio":
		welcome_text = "Hi There, click on the voice recorder to interact with me, How can I assist you today?"
		st.write (welcome_text)
		tts(welcome_text,'en')
		transcription = audio_processing()
		# transcription= audio_to_text("audio.mp3")
		# st.write("User:",query)
		if transcription:
			query = transcription.text
			answer = rag(query)
			add_to_history(query, answer)
			tts(answer,'en')
	elif option == "Chat":
		st.write("Wecome to text chatbot")
		with st.container():
			st.subheader("Your Input")
			query =st.text_input("Ask me anything:", key="user_input", on_change=None)
			# query=st.text_input("Search the topic u want", placeholder="Enter your query here...")
			if query:
				answer= rag(query)
				add_to_history(query, answer)
	else:
		st.write("Select your mode of interaction Chat/Audio")
	

#Function to transcribe audio to text 
if __name__ == "__main__" :
	main()

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from groq import Groq

groq_api_key = st.secrets["GROC_API_KEY"]


# Initialize the Groq client
client = Groq(api_key=groq_api_key)

def audio_to_text(audio_file):
	# Open the audio file
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
	    
def audio_recording():
	recorded_audio = audio_recorder()
	if recorded_audio:
		audio_file = "audio.mp3"
		with open(audio_file , "wb") as f:
			f.write(recorded_audio)

def page_layout():
	st.sidebar.title("Select the Modality")
	option = st.sidebar.selectbox(
	    "How would you like to be interact?",
	    ("Chat", "Audio"),
		index=None,
	     placeholder="Select mode of communication..."
	)
	st.title (":blue[ANCIENT GREEK Q&A CHATBOT] ")
	return option

def main():
	option = page_layout()
	if option == "Audio":
		st.write ("Hi There, click on the voice recorder to interact with me, How can I assist you today?")
		audio_recording()
		transcription= audio_to_text("audio.mp3")
		query = transcription.text
		st.write("User:",query)
		
	elif option == "Chat":
		st.write("Wecome to text chatbot")
		query=st.text_input("Search the topic u want", placeholder="Enter your query here...")
		st.write("User:",query)
	else:
		st.write("Select your mode of interaction Chat/Audio")

#Function to transcribe audio to text 

if __name__ == "__main__" :
	main()



import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from groq import Groq

groq_api_key = st.secrets["GROC_API_KEY"]


# Initialize the Groq client
client = Groq(api_key=groq_api_key)


def main():
	st.sidebar.title("Select the Modality")
	option = st.selectbox(
	    "How would you like to be interact?",
	    ("Chat", "Audio"),
	)
	st.title ("ANCIENT GREEK Q&A blue[CHATBOT] :sunglasses:")
	st.write ("Hi There, click on the voice recorder to interact with me, How can I assist you today?")
	recorded_audio = audio_recorder()
	if recorded_audio:
		audio_file = "audio.mp3"
		with open(audio_file , "wb") as f:
			f.write(recorded_audio)
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
		    # Print the transcription text
		    st.write("User:",transcription.text)

#Function to transcribe audio to text 

if __name__ == "__main__" :
	main()



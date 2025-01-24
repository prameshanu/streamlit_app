import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from groq import Groq

groq_api_key = st.secrets["GROC_API_KEY"]


def main():
	st.sidebar.title("API KEY CONFIGURATION")
	st.title ("Audio Test")
	st.write ("Hi There, click on the voice recorder to interact with me, How can I assist you today?")
	st.write(groq_api_key)
	recorded_audio = audio_recorder()

#Function to transcribe audio to text 

if __name__ == "__main__" :
	main()

# # Initialize the Groq client
# client = Groq()

# # Specify the path to the audio file
# filename = os.path.dirname(__file__) + "/sample_audio.m4a" # Replace with your audio file!

# # Open the audio file
# with open(filename, "rb") as file:
#     # Create a translation of the audio file
#     translation = client.audio.translations.create(
#       file=(filename, file.read()), # Required audio file
#       model="whisper-large-v3", # Required model to use for translation
#       prompt="Specify context or spelling",  # Optional
#       response_format="json",  # Optional
#       temperature=0.0  # Optional
    # )
    # Print the translation text
    # print(translation.text)

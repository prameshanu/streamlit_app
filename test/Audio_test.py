import streamlit as st
from audio_recorder_streamlit import audio_recorder



def main():
	st.sidebar.title("API KEY CONFIGURATION")
	st.title ("Audio Test")
	st.write ("Hi There, click on the voice recorder to interact with me, How can I assist you today?")
	recorded_audio = audio_recorder()

if __name__ == "__main__" :
	main()
	

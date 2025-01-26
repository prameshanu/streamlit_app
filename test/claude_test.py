from streamlit_extras.stylable_container import stylable_container
from audio_recorder_streamlit import audio_recorder
import streamlit as st
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder
import os
from groq import Groq
import numpy as np


groq_api_key = st.secrets["GROC_API_KEY"]
pine_cone_api_key = st.secrets["PINE_CONE_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]


# Initialize the Groq client
client = Groq(api_key=groq_api_key)


def tts(text_to_read, language):
	aud_file = gTTS(text=text_to_read, lang=language, slow=False,tld='co.in')
	aud_file.save("lang.mp3")
	audio_file_read = open('lang.mp3', 'rb')
	audio_bytes = audio_file_read.read()
	st.audio(audio_bytes, format='audio/mp3',autoplay=True)


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


title = "ANCIENT GREEK Q&A CHATBOT"



def example():
	with stylable_container(
		key="green_button",
		css_styles="""
  		button {
                /* background-color: green; */
                color: white;
		border: none;  
                border-radius: 0;
            	}
            	""",
    	):
		st.button("Dummy button")

	with stylable_container(
		key="green_button_a",
		css_styles="""
  		button {
                /* background-color: green; */
                color: white;
		border: none;  
                border-radius: 0;
            	}
            	""",
    	):
		st.button("Dummy button2")
## Heading and option button
	    
	with stylable_container(
        key="heading",
        css_styles="""
            	{
                position: fixed;
                top: 2%; 
                width: 80%;
                left: 10%; 
                right: 0;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem 0.5rem 0 0;
                padding: 1em;
                background-color: white;
                z-index: 100;
            	}
        	""",
	):
			
        # Apply custom CSS for full-width input
	
		st.markdown(
			"""
			<style>
			.full-width-input .stTextInput > div > div {
			width: 100%; /* Slightly smaller width */
			margin: 0 auto; /* Center it with equal margins on both sides */                
			}
			</style>
			""",
			unsafe_allow_html=True,
		)
    
        # Wrap the text input in a class to target it
	
		with st.container():
			st.title(f""":blue[{title}]""")
			option = st.selectbox(
				"How would you like to be interact?",
				("Chat","Audio"),
				index= None,
				placeholder = "Select mode of communication.."
			)
    
        # Wrap the text input in a class to target it
		# w
	  #       with st.container():
			# st.write(f""":blue{title}""")
			# # st.title(f""":blue[{title}] """)
			# option = st.selectbox(
			# 	"How would you like to be interact?",
			# 	("Chat", "Audio"),
	  #           		index=None,
	  #                	placeholder="Select mode of communication..."
	  #           		)

## Bottom input bar
	with stylable_container(
        	key="container_with_border",
        	css_styles="""
            	{
                	position: fixed;
	                bottom: 0; 
	                width: 80%;
	                left: 10%; 
	                right: 0;
	                border: 1px solid rgba(49, 51, 63, 0.2);
	                border-radius: 0.5rem 0.5rem 0 0;
	                padding: 1em;
	                background-color: white;
	                z-index: 100;
		}
   		""",
    	):
			
        # Apply custom CSS for full-width input
	
		st.markdown(
			"""
			<style>
			.full-width-input .stTextInput > div > div {
			width: 100%; /* Slightly smaller width */
			margin: 0 auto; /* Center it with equal margins on both sides */                
			}
			</style>
			""",
			unsafe_allow_html=True,
		)
    
        # Wrap the text input in a class to target it
		
		with st.container():
			if option == "Audio": 
				welcome_text = "Hi There, click on the voice recorder to interact with me, How can I assist you today?"
				st.write (welcome_text)
				tts(welcome_text,'en')
				transcription = audio_processing()
				# transcription= audio_to_text("audio.mp3")
				# st.write("User:",transcription.text)
				
				if transcription:
					query = transcription.text
				else:
					query = 'No recorded voice'
				values = np.array([option, query])
				return values
			elif option == "Chat":
				query = st.text_input(
					"Type your message here:",
					key="user_input",
					label_visibility="collapsed",
					placeholder="Type your message...",
				)
				values = np.array([option, query])
				return values
				# return query,option




# values = []



# st.button("")
text = "this is beta testing"
def write_function(text):
	st.markdown(f"""<p style="position: fixed; width: 80%; left: 11%; right: 0;">{text}</p>""", unsafe_allow_html=True)


values = np.array([])
values = example()
# st.write("\n.........................................Dummy.........................................\n")
if values is None:
	write_function("Kindly select the mode of communication from above drop-down button")
elif len(values) > 0 and values[1] == "":
	write_function("Please write below your query")
else:
	# write_function(f"**User:** {values[1]}")
	st.write(f"**User:** {values[1]}")
# st.write(len(values))
# if len(values) == 0:
# 	st.write("Enter modality")
# elif values[1]:	
# 	query = values[1]
# 	st.write("Test")
# 	st.write(f"**User:** {query}")
# 	st.write(values[0])
# else:
# 	st.write("Test")

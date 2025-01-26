from streamlit_extras.stylable_container import stylable_container
from audio_recorder_streamlit import audio_recorder
import streamlit as st

title = "ANCIENT GREEK Q&A CHATBOT"
option = "Chat"
def example():
	with stylable_container(
		key="green_button",
		css_styles="""
  		button {
                background-color: green;
                color: white;
                border-radius: 20px;
            	}
            	""",
    	):
		st.button("Green button")

## Heading and option button
	    
	with stylable_container(
        key="heading",
        css_styles="""
            	{
                position: fixed;
                top: 10%; 
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
				st.write("Audio testing") 
			elif option == "Chat":
				st.text_input(
					"Type your message here:",
					key="user_input",
					label_visibility="collapsed",
					placeholder="Type your message...",
				)


example()

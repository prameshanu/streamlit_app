import streamlit as st

st.title("Simple Chatbot")

def chatbot_response(message):
    responses = {
        "hello": "Hi there! How can I assist you?",
        "bye": "Goodbye! Have a nice day!",
    }
    return responses.get(message.lower(), "Sorry, I don't understand.")

user_message = st.text_input("You:", "")
if user_message:
    bot_reply = chatbot_response(user_message)
    st.write(f"Bot: {bot_reply}")

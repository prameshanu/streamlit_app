
import streamlit as st
from groq import Groq

groq_api_key = st.secrets["GROC_API_KEY"]

# Initialize the Groq client
client = Groq(api_key=groq_api_key)



st.title("ChatGPT-like clone")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [
            {
                "role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
        ]
    
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        response = completion.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})

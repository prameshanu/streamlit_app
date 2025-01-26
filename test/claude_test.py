import streamlit as st

# Inject custom CSS to make the title fixed
st.markdown(
    """
    <style>
    .fixed-title {
        position: fixed;
        top: 40px;
        left = 50px;
        width: 100%;
        background-color: #ffffff;
        z-index: 100;
        padding: 10px 0;
        box-shadow: 0px 0px 0px rgba(0,0,0,0.1);
        text-align: left;
    }
    .content {
        margin-top: 60px; /* Adjust this based on the title height */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fixed title
st.markdown('<div class="fixed-title"><h1>Chatbot with Persistent History</h1></div>', unsafe_allow_html=True)

# Main content area
st.markdown('<div class="content">', unsafe_allow_html=True)

# Initialize session state for chat history if not already present
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to render chat history
def render_chat_history():
    for chat in st.session_state["chat_history"]:
        user_query, bot_response = chat
        st.write(f"**User:** {user_query}")
        st.write(f"**Bot:** {bot_response}")

# Function to add a message to the chat history
def add_to_history(user_query, bot_response):
    st.session_state["chat_history"].append((user_query, bot_response))

# Render chat history
st.subheader("Chat History")
render_chat_history()

# Place the input bar at the bottom
with st.container():
    st.subheader("Your Input")
    user_input = st.text_input("Ask me anything:", key="user_input", on_change=None)

    # Process user input
    if user_input:
        # Example bot response logic
        bot_response = f"Here is the answer to: {user_input}"  # Replace with actual response logic
        add_to_history(user_input, bot_response)
        # st.experimental_rerun()  # Refresh the app to show updated chat history

st.markdown('</div>', unsafe_allow_html=True)

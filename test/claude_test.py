import streamlit as st

# Initialize session state for chat history if not already present
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to render chat history in the sidebar
def render_chat_history():
    # Display chat history as selectable options in the sidebar
    if st.session_state["chat_history"]:
        chat_choices = [f"Conversation {i+1}" for i in range(len(st.session_state["chat_history"]))]
        selected_chat_index = st.sidebar.selectbox("Select a conversation", chat_choices)

        # Get the selected chat history based on the index
        selected_chat = st.session_state["chat_history"][chat_choices.index(selected_chat_index)]
        return selected_chat
    return None

# Function to add a message to the chat history
def add_to_history(user_query, bot_response):
    st.session_state["chat_history"].append((user_query, bot_response))

# Render title in the main area
st.title("Chatbot with Persistent History")

# Place the input bar at the bottom
with st.container():
    st.subheader("Your Input")
    user_input = st.text_input("Ask me anything:", key="user_input", on_change=None)

    # Process user input
    if user_input:
        # Example bot response logic
        bot_response = f"Here is the answer to: {user_input}"  # Replace with actual response logic
        add_to_history(user_input, bot_response)

# Render chat history in the sidebar and select the conversation
selected_chat = render_chat_history()

# Display selected chat in the main area
if selected_chat:
    for user_query, bot_response in selected_chat:
        st.write(f"**User:** {user_query}")
        st.write(f"**Bot:** {bot_response}")

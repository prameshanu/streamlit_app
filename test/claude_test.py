import streamlit as st

# Initialize session state for text input if it doesn't already exist
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Function to handle the submit button and reset the text input
def submit_text():
    entered_text = st.session_state.text_input
    st.write(f"You entered: {entered_text}")
    # Reset text input by modifying session state before widget is instantiated
    st.session_state.text_input = ""

# Create a horizontal layout using columns
col1, col2 = st.columns([4, 1])

# Place the text input in the first column
with col1:
    st.text_input(
        "Enter your text here:",
        key="text_input",  # Bind it to session state
    )

# Place the submit button in the second column
with col2:
    if st.button("Submit"):
        submit_text()

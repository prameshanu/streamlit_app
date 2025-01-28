import streamlit as st

# Initialize session state for the text input
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Function to handle form submission
def submit_text():
    entered_text = st.session_state.text_input
    st.write(f"You entered: {entered_text}")
    # Reset the text input to default
    st.session_state.text_input = ""

# Create columns for horizontal layout
col1, col2 = st.columns([4, 1])  # Adjust the ratio to control the size of each column

# Place the text input in the first column
with col1:
    st.text_input(
        "Enter your text here:",
        key="text_input",
    )

# Place the submit button in the second column
with col2:
    if st.button("Submit"):
        submit_text()

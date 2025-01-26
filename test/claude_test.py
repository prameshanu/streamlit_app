

import streamlit as st

# Add custom CSS to make the input field sticky
st.markdown(
    """
    <style>
    /* Make the footer container sticky */
    .footer-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white; /* Adjust background to match your app */
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1); /* Optional shadow */
        z-index: 1000; /* Ensure it stays above other elements */
    }
    .footer-container input {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main content
st.title("Ask Me Anything")
st.write("This is your interactive assistant. Type your query below!")

# Sticky footer container for input
st.markdown(
    """
    <div class="footer-container">
        <form action="" method="GET">
            <input type="text" id="user_input" name="user_input" placeholder="Ask me anything...">
        </form>
    </div>
    """,
    unsafe_allow_html=True,
)

# Read user input from the URL query parameter (if needed)
# user_query = st.experimental_get_query_params().get("user_input", [""])[0]
user_query = st.query_params().get("user)input",[""])[0]

# Display the query if submitted
if user_query:
    st.write(f"You asked: {user_query}")

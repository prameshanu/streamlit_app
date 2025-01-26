import streamlit as st

# Add custom CSS for a sticky footer input box that adjusts with the sidebar
st.markdown(
    """
    <style>
    /* Sticky footer container */
    .footer-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 200%; 
        /* background-color: white; */
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        display: flex;
        justify-content: left;
    }
    
    /* Center-aligned flexible input box */
    .footer-container input {
        width: 100%; /* Flexible width */
        max-width: 800px; /*  Optional: cap the max width */
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }
    
    /* Adjust the app content for the footer */
    .stApp {
        padding-bottom: 70px; /* Space for the footer */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar (optional)
st.sidebar.title("Sidebar")
st.sidebar.write("Add your sidebar content here.")

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

# Read user input from the URL query parameter
user_query = st.query_params.get("user_input", "")

# Display the query if submitted
if user_query:
    st.write(f"You asked: {user_query}")

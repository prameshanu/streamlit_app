import streamlit as st

# Add custom CSS for a sticky footer input box that adjusts with sidebar
st.markdown(
    """
    <style>
    /* Sticky footer container */
    .footer-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: calc(100% - var(--sidebar-width)); /* Adjusts for sidebar width */
        background-color: white; /* Match the app's background */
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1); /* Optional shadow */
        z-index: 1000; /* Ensure it stays above other elements */
    }
    
    /* Responsive input styling */
    .footer-container input {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }
    
    /* Sidebar-aware adjustments */
    .stApp {
        padding-bottom: 70px; /* Ensure space for the footer */
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

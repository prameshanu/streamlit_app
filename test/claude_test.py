import streamlit as st

# Add custom CSS to make the input field dynamic and responsive to sidebar
st.markdown(
    """
    <style>
    /* Make the input field container sticky at the bottom */
    .footer-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: calc(100% - var(--sidebar-width)); /* Dynamic width adjustment */
        margin-left: var(--sidebar-width);
        background-color: white;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }

    /* Sidebar width handling */
    [data-testid="stSidebar"] {
        --sidebar-width: 300px; /* Approximate sidebar width */
    }

    /* Style the input box */
    .footer-container input {
        width: 100%; /* Full width of available space */
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }

    /* Adjust the app content to make space for the footer */
    .stApp {
        padding-bottom: 70px; /* Space for the footer */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Sidebar")
st.sidebar.write("This is the sidebar content.")

# Main content
st.title("Dynamic Input Field with Sidebar")
st.write("This input field at the bottom will dynamically resize based on the sidebar's presence.")

# Sticky footer input container
st.markdown(
    """
    <div class="footer-container">
        <form action="" method="GET">
            <input type="text" id="user_input" name="user_input" placeholder="Type your message here...">
        </form>
    </div>
    """,
    unsafe_allow_html=True,
)

# Handle the query if input is provided
user_query = st.query_params.get("user_input", "")
if user_query:
    st.write(f"You asked: {user_query}")

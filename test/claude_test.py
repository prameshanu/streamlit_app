from streamlit_extras.stylable_container import stylable_container
import streamlit as st

def example():
    with stylable_container(
        key="green_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
            }
            """,
    ):
        st.button("Green button")

    st.button("Normal button")

    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                position: fixed;
                bottom: 0; 
                width: 80%;
                left: 10%; 
                right: 0;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem 0.5rem 0 0;
                padding: 1em;
                background-color: white;
                z-index: 100;
            }
        """,
    ):
        # Apply custom CSS for full-width input
        st.markdown(
            """
            <style>
            .full-width-input .stTextInput > div > div {
                width: 40%; /* Slightly smaller width */
                margin: 0 auto; /* Center it with equal margins on both sides */
                
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    
        # Wrap the text input in a class to target it
        with st.container():
            st.text_input(
                "Type your message here:",
                key="user_input",
                label_visibility="collapsed",
                placeholder="Type your message...",
            )




example()

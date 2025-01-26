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
                left: 0;
                right: 0;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px);
                background-color: white; /* Optional: Set a background color to make it visible */
                z-index: 100; /* Optional: Ensure it stays above other elements */
            }
            """,
    ):


    # with stylable_container(
    #     key="container_with_border",
    #     css_styles="""
    #         {
    #             top: 100px;
    #             border: 1px solid rgba(49, 51, 63, 0.2);
    #             border-radius: 0.5rem;
    #             padding: calc(1em - 1px)
    #         }
    #         """,
    # ):
        st.markdown("This is a container with a border.")


example()

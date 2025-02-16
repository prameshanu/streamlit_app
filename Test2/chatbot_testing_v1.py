from pinecone import Pinecone
import streamlit as st
pine_cone_api_key = st.secrets["PINE_CONE_API_KEY"]
pc = Pinecone(api_key=pine_cone_api_key)
# List available indexes to verify connectivity
st.write(pc.list_indexes().names())


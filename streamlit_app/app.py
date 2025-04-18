
import streamlit as st
import requests

st.title("LLM-based RAG Search")
query = st.text_input("Enter your query:")

if st.button("Search") and query:
    response = requests.post("http://localhost:5001/query", json={"query": query})
    if response.status_code == 200:
        answer = response.json().get('answer', "No answer received.")
        st.write("Answer:", answer)
    else:
        st.error(f"Error: {response.status_code}")

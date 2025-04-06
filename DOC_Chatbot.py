"""
------------------------RAG-Based Document Chatbot using Google Gemini and Streamlit--------------------------------

This chatbot allows users to upload a single PDF document and query using Retrieval-Augmented Generation (RAG).
It extracts text from the document, chunks it, finds relevant context, and then queries Google Gemini for responses.
"""

# -------------------------------------IMPORTING ESSENTIALS-------------------------------------------

import fitz
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ---------------------------------CONFIGURATION USING GEMINI API-------------------------------------

genai.configure(api_key="AIzaSyD0n9mntLuB_qFPZsbPaUB8Pry2OydswLU")

# -------------------------------------PDF TEXT EXTRACTION---------------------------------------------

def extract_text_from_pdf(pdf_path):
    #Extracts text from a given PDF file.
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# ----------------------------------------TEXT CHUNKING -----------------------------------------------

def split_text_into_chunks(text, chunk_size=500):
    # Splits extracted text into manageable chunks for retrieval.
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# --------------------------------------EMBEDDING CREATION ---------------------------------------------

def create_embeddings(chunks):
    # Creates embeddings using TF-IDF vectorization.
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks)
    return embeddings, vectorizer

# --------------------------------------RETRIEVAL FUNCTION ---------------------------------------------

def retrieve_relevant_chunk(query, chunks, vectorizer, embeddings):
    # Finds the most relevant text chunk for a given query.
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, embeddings)
    best_match_index = np.argmax(similarities)
    return chunks[best_match_index]

# ----------------------------------------GEMINI API CALL ----------------------------------------------

def ask_gemini(query, context=""):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(f"Context:\n{context}\n\nQuery:\n{query}\n\nProvide a structured response.")
    return response.text

# ------------------------------------------STREAMLIT UI------------------------------------------------

st.title("📄 RAG-Based Document Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""

doc_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if doc_file:
    with open("temp.pdf", "wb") as f:
        f.write(doc_file.read())
    
    # Processing the document
    text = extract_text_from_pdf("temp.pdf")
    chunks = split_text_into_chunks(text)
    embeddings, vectorizer = create_embeddings(chunks)

    # Changing button color
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: black;
        color: white;
    }
    </style>
     """, unsafe_allow_html=True)

    
    # Displaying chat history
    st.subheader("📝Chat History:")
    for q, a in st.session_state.chat_history:
        st.write(f"**YOU🙋‍♀️:** {q}")
        st.write(f"**BOT🤖:** {a}")
    with st.form("chat_form", clear_on_submit=True):
        placeholder_text = "Ask a question..." if not st.session_state.chat_history else "Ask another question..."
        query = st.text_input("Type your message here:",placeholder=placeholder_text,  key="query")
        submit_button = st.form_submit_button("Send")

    if submit_button:
        if query:
            context = retrieve_relevant_chunk(query, chunks, vectorizer, embeddings)
            response = ask_gemini(query, context)
            st.session_state.chat_history.append((query, response))
            st.rerun()
        else:
            st.warning("Please enter a valid query.")
            
# --------------------------------------------------------------------------------------------------------
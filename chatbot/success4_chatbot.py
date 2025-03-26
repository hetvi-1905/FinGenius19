# uploading mutliple docs by user, and better text extraction to get better output 

import streamlit as st
import os
import google.generativeai as genai
import faiss
import numpy as np
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load API Keys
# try:
#     GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
#     GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# except Exception as e:
#     st.error(f"⚠️ Error: Could not load API Keys. Check `secrets.toml`. Details: {e}")
import os
import dotenv  # Import dotenv package

# Load API Keys from .env file
dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Check if API keys are loaded correctly
if not GROQ_API_KEY or not GEMINI_API_KEY:
    st.error("⚠️ Missing API Keys. Please check your .env file.")
# Initialize Gemini AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("⚠️ Missing Google Gemini API Key.")

# Initialize FAISS Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None  # Initialize FAISS storage
raw_texts = []  # Store raw text for fallback search

# Function to process uploaded documents
def process_uploaded_documents(uploaded_files):
    """Extracts and indexes text from multiple uploaded documents."""
    global raw_texts
    all_chunks = []
    raw_texts = []  # Reset raw text storage

    for uploaded_file in uploaded_files:
        temp_path = f"temp_uploaded.{uploaded_file.name.split('.')[-1]}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_path)
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
            continue

        docs = loader.load()
        
        # Store raw text for fallback search
        for doc in docs:
            raw_texts.append(doc.page_content)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

    if all_chunks:
        return FAISS.from_documents(all_chunks, embeddings)
    return None

# Function to retrieve financial info from FAISS
def retrieve_financial_info(query, vectorstore):
    """Search FAISS index for relevant financial documents."""
    if not vectorstore:
        return ["No document uploaded."]
    
    query_embedding = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(np.array(query_embedding), k=3)
    
    retrieved_texts = [doc.page_content for doc in results] if results else []

    # If FAISS fails, fallback to simple text search
    if not retrieved_texts:
        retrieved_texts = [text for text in raw_texts if query.lower() in text.lower()]

    return retrieved_texts if retrieved_texts else ["No relevant finance documents found."]

# Function to get AI response from Gemini
def get_gemini_response(prompt):
    """Fetches AI-generated response from Gemini API."""
    if not GEMINI_API_KEY:
        return "Error: Missing Gemini API Key"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("💰 AI-Powered Finance Chatbot with Document Upload")

# File Upload Section
uploaded_files = st.file_uploader("Upload financial documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    vectorstore = process_uploaded_documents(uploaded_files)
    if vectorstore:
        st.success("🔍 Documents processed and indexed for retrieval!")
    else:
        st.error("⚠️ Failed to process documents.")

# User Query Section
user_input = st.text_input("Ask me anything about finance:")
if user_input:
    retrieved_info = retrieve_financial_info(user_input, vectorstore)
    
    full_prompt = f"""
    Financial query: {user_input}
    
    Retrieved Document Insights:
    {retrieved_info}
    
    Ensure the response contains factual information from the documents. If the document data is insufficient, provide the best answer based on knowledge.
    """

    response = get_gemini_response(full_prompt)

    st.write("### 🤖 AI Response:")
    st.write(response)


# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success4_chatbot.py"


# Little improve results than success3

# 💰 AI-Powered Finance Chatbot with Document Upload
# Upload financial documents (PDF, DOCX, TXT)

# Annual-Report_Suyog_2024.pdf
# Drag and drop files here
# Limit 200MB per file • PDF, DOCX, TXT
# Annual-Report_Suyog_2024.pdf
# 8.3MB
# 🔍 Documents processed and indexed for retrieval!

# Ask me anything about finance:

# who is the managing director of suyog telematics?
# 🤖 AI Response:
# Based on the provided document, the Managing Director of Suyog Telematics Limited is Shivshankar Lature. His DIN (Director Identification Number) is 02090972.
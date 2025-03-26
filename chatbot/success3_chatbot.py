# doc uploading by user and getting response 
import streamlit as st
import yfinance as yf
import requests
import os
import google.generativeai as genai
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# Load API Keys from Streamlit secrets
# try:
#     GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
#     NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
#     PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
#     GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# except Exception as e:
#     st.error(f"⚠️ Error: Could not load API Keys. Make sure `secrets.toml` is in the correct path.\nDetails: {e}")
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
    st.error("⚠️ Error: Missing Google Gemini API Key.")

# Initialize FAISS
dimension = 768  
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

# Load Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Updated model

# Function to process uploaded documents
def process_uploaded_document(uploaded_file):
    """Loads a PDF, DOCX, or TXT file and extracts text."""
    temp_path = f"temp_uploaded.{uploaded_file.name.split('.')[-1]}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())  # Save the uploaded file locally

    # Load document based on file type
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(temp_path)
    else:
        return None, "⚠️ Unsupported file type!"
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    return chunks, "✅ Document successfully loaded!"

# Function to retrieve finance-related documents from FAISS
def retrieve_financial_info(query, vectorstore):
    """Search FAISS vector store for relevant finance documents."""
    query_embedding = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(np.array(query_embedding), k=3)
    
    return [doc.page_content for doc in results] if results else ["No relevant finance documents found."]

# Function to get AI response from Groq
def get_groq_response(prompt):
    """Fetches AI-generated response from Groq API."""
    if not GROQ_API_KEY:
        return "Error: Missing Groq API Key"
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt}]}

    response = requests.post("https://api.groq.com/v1/chat/completions", json=data, headers=headers)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")

# Function to get AI response from Gemini
def get_gemini_response(prompt):
    """Fetches AI-generated response from Gemini API."""
    if not GEMINI_API_KEY:
        return "Error: Missing Gemini API Key"
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Free model
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("💰 AI-Powered Finance Chatbot with Document Upload")

# File Upload Section
uploaded_file = st.file_uploader("Upload a financial document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

vectorstore = None  # Placeholder for FAISS vector store
if uploaded_file:
    documents, status = process_uploaded_document(uploaded_file)
    st.write(status)

    if documents:
        vectorstore = FAISS.from_documents(documents, embeddings)
        st.success("🔍 Document processed and indexed for retrieval!")

# User Query Section
user_input = st.text_input("Ask me anything about finance:")

if user_input:
    # Retrieve relevant finance info
    retrieved_info = retrieve_financial_info(user_input, vectorstore) if vectorstore else ["No document uploaded."]
    
    # Generate AI response using Gemini
    full_prompt = f"Financial query: {user_input}\nRelevant info: {retrieved_info}"
    response = get_gemini_response(full_prompt)

    st.write("### 🤖 AI Response:")
    st.write(response)

# Display Latest Financial News
def get_financial_news():
    """Fetches top financial news from News API."""
    if not NEWS_API_KEY:
        return [("Error: Missing News API Key", "#")]
    
    url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [(article["title"], article["url"]) for article in articles[:5]]
    except Exception as e:
        return [(f"Error: {str(e)}", "#")]

st.write("### 📰 Latest Financial News")
for title, url in get_financial_news():
    st.write(f"[{title}]({url})")

# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success3_chatbot.py"

# bas results in text extraction from doc:
# 💰 AI-Powered Finance Chatbot with Document Upload
# Upload a financial document (PDF, DOCX, TXT)

# Annual-Report_Suyog_2024.pdf
# Drag and drop file here
# Limit 200MB per file • PDF, DOCX, TXT
# Annual-Report_Suyog_2024.pdf
# 8.3MB
# ✅ Document successfully loaded!

# 🔍 Document processed and indexed for retrieval!

# Ask me anything about finance:

# who is the managing director of suyog?
# 🤖 AI Response:
# The provided text doesn't state who the Managing Director of Suyog is. While it mentions Ajay Kumar Banwarilal Sharma as the Chief Financial Officer and refers to a Managing Director signing a document, the name of the Managing Director is not explicitly given.

# 📰 Latest Financial News
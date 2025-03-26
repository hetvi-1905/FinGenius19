# toggling between 2 chatbots: 1) general QA on finance 2) doc uploading + QA related to the doc and both grok + gemini response side by side 

import streamlit as st
import google.generativeai as genai
import groq
import yfinance as yf
import requests
import faiss
import numpy as np
import pdfplumber
import pytesseract
import fitz  # PyMuPDF for text extraction
import plotly.express as px
from PIL import Image
from io import BytesIO
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

import pytesseract

# Specify the Tesseract path (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load API Keys
# try:
#     GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
#     GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
#     NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
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


# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)
groq_client = groq.Client(api_key=GROQ_API_KEY)

# Initialize FAISS Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None
raw_texts = []

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text, tables, and images from a PDF file."""
    extracted_text = ""

    # Extract text using PyMuPDF
    doc = fitz.open(pdf_path)
    for page in doc:
        extracted_text += page.get_text("text") + "\n"

        # Extract images and apply OCR
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image)
            extracted_text += f"\n[Extracted from Image {img_index}]\n{ocr_text}\n"

    # Extract tables using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    extracted_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"

    return extracted_text


# Function to process uploaded documents
def process_uploaded_documents(uploaded_files):
    """Extracts and indexes text from multiple uploaded documents."""
    global raw_texts
    all_chunks = []
    raw_texts = []

    for uploaded_file in uploaded_files:
        temp_path = f"temp_uploaded.{uploaded_file.name.split('.')[-1]}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        extracted_text = ""
        if uploaded_file.name.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
            extracted_text = " ".join([doc.page_content for doc in loader.load()])
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_path)
            extracted_text = " ".join([doc.page_content for doc in loader.load()])
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
            continue

        raw_texts.append(extracted_text)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)

        # Convert to LangChain Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]
        all_chunks.extend(documents)

    if all_chunks:
        return FAISS.from_documents(all_chunks, embeddings)
    return None


# Function to retrieve document info
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

# Function to get responses from AI models
def get_groq_response(prompt):
    """Fetches AI-generated response from Groq API."""
    if not GROQ_API_KEY:
        return "⚠️ Error: Missing Groq API Key"

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}"

def get_gemini_response(prompt):
    """Fetches AI-generated response from Gemini API."""
    if not GEMINI_API_KEY:
        return "⚠️ Error: Missing Gemini API Key"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        return f"⚠️ Gemini API Error: {str(e)}"

# Streamlit UI
st.title("💰 AI Finance Chatbot (Groq + Gemini)")
mode = st.radio("Select Mode:", ["General Finance Chat", "Document-Based Analysis"])

if mode == "General Finance Chat":
    user_query = st.text_input("Ask me anything about finance:")
    if user_query:
        groq_response = get_groq_response(user_query)
        gemini_response = get_gemini_response(user_query)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 Groq (LLaMA 3)")
            st.write(groq_response)
        with col2:
            st.subheader("🔍 Gemini (Google)")
            st.write(gemini_response)

if mode == "Document-Based Analysis":
    uploaded_files = st.file_uploader("Upload financial documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        vectorstore = process_uploaded_documents(uploaded_files)
        if vectorstore:
            st.success("📂 Documents indexed for retrieval!")
        else:
            st.error("⚠️ Failed to process documents.")
    user_input = st.text_input("Ask a question related to uploaded documents:")
    # if user_query and vectorstore:
    #     retrieved_info = retrieve_financial_info(user_query, vectorstore)
    #     st.write("📄 **Extracted Document Data:**")
    #     st.write(retrieved_info)
    if user_input:
          retrieved_info = retrieve_financial_info(user_input, vectorstore)

          full_prompt = f"""
          Financial query: {user_input}
    
          Retrieved Document Insights:
          {retrieved_info}
    
          Ensure the response contains factual information from the documents. If the document data is insufficient, provide the best answer based on knowledge.
           """

          with st.spinner("Fetching responses..."):
               groq_response = get_groq_response(full_prompt)
               gemini_response = get_gemini_response(full_prompt)

    # Display responses side by side
          col1, col2 = st.columns(2)

          with col1:
                st.subheader("🤖 Groq (LLaMA 3)")
                st.write(groq_response)

          with col2:
                st.subheader("🔍 Gemini (Google)")
                st.write(gemini_response)


# Real-time Stock Market Data
st.subheader("📊 Real-Time Stock Market Data")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS):")
if ticker:
    stock = yf.Ticker(ticker)
    info = stock.info
    st.write(f"**{info['shortName']}**")
    st.write(f"📈 **Current Price:** Rs.{info.get('currentPrice', 'N/A')}")

# Interactive Stock Performance Chart
if ticker:
    hist = stock.history(period="1mo")
    fig = px.line(hist, x=hist.index, y="Close", title=f"{ticker} Stock Prices")
    st.plotly_chart(fig)

# AI-Powered News Sentiment Analysis
st.subheader("📰 Latest Finance News")
url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}"
response = requests.get(url).json()
news = response.get("articles", [])[:5]
for article in news:
    st.write(f"🔹 [{article['title']}]({article['url']})")

# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success9_chatbot.py"
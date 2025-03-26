import streamlit as st
import os
import google.generativeai as genai
import faiss
import numpy as np
import requests
import fitz  # PyMuPDF for better text extraction
import pytesseract  # OCR for images
import pdfplumber  # Extracts tables from PDFs
from PIL import Image
from io import BytesIO
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pytesseract

# Specify the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows

# # Load API Keys
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

# Function to extract text, images, and tables from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text, tables, and images from a PDF file."""
    extracted_text = ""

    # Extract text using PyMuPDF (better accuracy)
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
            tables = page.extract_table()
            if tables:
                for row in tables:
                    # Convert NoneType cells to empty strings
                    extracted_text += " | ".join(str(cell) if cell is not None else "" for cell in row) + "\n"

    return extracted_text


# Function to process uploaded documents
from langchain_core.documents import Document

def process_uploaded_documents(uploaded_files):
    """Extracts and indexes text from multiple uploaded documents."""
    global raw_texts
    all_chunks = []
    raw_texts = []  # Reset raw text storage

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

        # Store raw text for fallback search
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
st.title("💰 AI-Powered Finance Chatbot with Advanced Document Extraction")

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
# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success5_chatbot.py"

# 🚀 Expected Results
# Ask: "What is the net profit ratio of Suyog in 2024?"
# Expected Response: "The net profit margin of Suyog Telematics in 2024 is 38.0%."

# Ask: "What is the EPS (Earnings Per Share) of Suyog in 2024?"
# Expected Response: "The EPS of Suyog Telematics in 2024 is 44.17 Rs. (both basic and diluted)."



# 💰 AI-Powered Finance Chatbot with Advanced Document Extraction
# Upload financial documents (PDF, DOCX, TXT)

# Annual-Report_Suyog_2024.pdf
# Drag and drop files here
# Limit 200MB per file • PDF, DOCX, TXT
# Annual-Report_Suyog_2024.pdf
# 8.3MB
# 🔍 Documents processed and indexed for retrieval!

# Ask me anything about finance:

# who is the managing director ?
# 🤖 AI Response:
# Based on the provided text, the Managing Director is Mr. Shivshankar Lature.

# 💰 AI-Powered Finance Chatbot with Advanced Document Extraction
# Upload financial documents (PDF, DOCX, TXT)

# Annual-Report_Suyog_2024.pdf
# Drag and drop files here
# Limit 200MB per file • PDF, DOCX, TXT
# Annual-Report_Suyog_2024.pdf
# 8.3MB
# 🔍 Documents processed and indexed for retrieval!

# Ask me anything about finance:

# What is the EPS (Earnings Per Share) of Suyog in 2024?
# 🤖 AI Response:
# Based on the provided text, the Earnings Per Share (EPS) for Suyog Telematics Limited for the financial year ended March 31, 2024,
#     is 44.17 Rs. (Basic) and 59.38 Rs. (Diluted). The document shows both basic and diluted EPS figures.

# 💰 AI-Powered Finance Chatbot with Advanced Document Extraction
# Upload financial documents (PDF, DOCX, TXT)

# No file chosen
# Drag and drop files here
# Limit 200MB per file • PDF, DOCX, TXT
# Annual-Report_Suyog_2024.pdf
# 8.3MB
# 🔍 Documents processed and indexed for retrieval!

# Ask me anything about finance:

# What is the net profit ratio of Suyog in 2024?
# 🤖 AI Response:
# The provided text gives conflicting information regarding Suyog's net profit in 2024.

# One section shows a net profit of ₹ 6,331.24 (presumably in Lakhs or Crores, the units are not specified). 
# Another section mentions an average net profit of ₹ 52,65,02,377 for a period, but doesn't specify if that 
# period includes 2024 or the length of the averaging period. Therefore, it's impossible to definitively 
# state Suyog's net profit ratio for 2024 without clarifying the units and the averaging period of the larger figure.
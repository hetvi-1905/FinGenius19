import streamlit as st
import yfinance as yf
import requests
import os
import google.generativeai as genai
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Load API Keys
# GROQ_API_KEY = os.getenv("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")  # Ensure it's set in the environment
# NEWS_API_KEY = os.getenv(" 67e2a7d22c5df8.27400249")  # Ensure it's set in the environment
# PINECONE_API_KEY = os.getenv("pcsk_3SJesG_CGqMZMznw8otmbc5A9iM46rXMzJ9QidpD2hpjtph3C3erZNVLkgPKcxAbv6S3cd")  # Ensure it's set in the environment
# GEMINI_API_KEY = os.getenv("AIzaSyBAjKwuIWqhLU1xd47mDolWRQxajLlzg84")  # Ensure it's set in the environment

# the key are in secrets.toml folder at : C:\Users\Admin\.streamlit __________________________________________________
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
embeddings = HuggingFaceEmbeddings()

# Function to retrieve finance-related documents
def retrieve_financial_info(query):
    query_embedding = np.random.rand(1, dimension).astype("float32")  # Replace with actual embedding
    if index.is_trained:
        D, I = index.search(query_embedding, k=3)
        return [f"Document {i+1}: Simulated finance info" for i in I[0]]
    else:
        return ["No relevant finance documents found."]

# Function to get AI response from Groq
def get_groq_response(prompt):
    if not GROQ_API_KEY:
        return "Error: Missing Groq API Key"
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt}]}

    response = requests.post("https://api.groq.com/v1/chat/completions", json=data, headers=headers)
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")

# Function to get AI response from Gemini (Using Free Model)
def get_gemini_response(prompt):
    if not GEMINI_API_KEY:
        return "Error: Missing Gemini API Key"
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Free model
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ROE": info.get("returnOnEquity", "N/A"),
            "Current Price": info.get("currentPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A")
        }
    except Exception as e:
        return {"Error": f"Could not retrieve stock data: {str(e)}"}

# Function to fetch financial news
def get_financial_news():
    if not NEWS_API_KEY:
        return [("Error: Missing News API Key", "#")]
    
    url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [(article["title"], article["url"]) for article in articles[:5]]
    except Exception as e:
        return [(f"Error: {str(e)}", "#")]

# Streamlit UI
st.title("💰 AI-Powered Finance Chatbot")

user_input = st.text_input("Ask me anything about finance:")

if user_input:
    # Retrieve relevant finance info
    retrieved_info = retrieve_financial_info(user_input)

    # Generate AI response using Gemini
    full_prompt = f"Financial query: {user_input}\nRelevant info: {retrieved_info}"
    response = get_gemini_response(full_prompt)

    st.write("### 🤖 AI Response:")
    st.write(response)

    # Stock insights section
    if "ROE" in user_input or "best stocks" in user_input:
        stock_symbol = st.text_input("Enter stock ticker (e.g., RELIANCE.NS, AAPL, TSLA):")
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol.upper())
            st.write(stock_data)

    # Financial news section
    st.write("### 📰 Latest Financial News")
    for title, url in get_financial_news():
        st.write(f"[{title}]({url})")

# To run the file: 
# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success1_chatbot.py"

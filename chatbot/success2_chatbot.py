# RAG integrated 
import streamlit as st
import yfinance as yf
import requests
import os
import google.generativeai as genai
import faiss
import numpy as np
import json
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------- Load API Keys --------------------
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
# -------------------- Initialize AI Models --------------------
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("⚠️ Error: Missing Google Gemini API Key.")

# -------------------- Initialize FAISS for RAG --------------------
dimension = 768  
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
embeddings = HuggingFaceEmbeddings()

# -------------------- Load Finance Knowledge Base --------------------
@st.cache_data
def load_financial_knowledge():
    """Load historical stock trends, market analysis, and reports."""
    knowledge_data = [
        "Reliance's ROE is 18.5% for 2024, showing strong profitability.",
        "TCS revenue increased by 10% in Q1 2024.",
        "NIFTY 50 index rose by 2% after positive economic growth data.",
        "Goldman Sachs predicts a 5% increase in US stocks for 2025.",
        "Tesla's stock volatility increased due to new AI innovations."
    ]
    embeddings_data = np.array([embeddings.embed_query(text) for text in knowledge_data], dtype="float32")
    index.add(embeddings_data)
    return knowledge_data

financial_knowledge = load_financial_knowledge()

# -------------------- Function: Retrieve Financial Insights using RAG --------------------
def retrieve_financial_info(query):
    """Search for the most relevant financial document using FAISS."""
    query_embedding = np.array([embeddings.embed_query(query)], dtype="float32")
    if index.ntotal == 0:
        return ["⚠️ No financial data available."]
    
    D, I = index.search(query_embedding, k=3)  # Get top 3 matches
    return [financial_knowledge[i] for i in I[0] if i < len(financial_knowledge)]

# -------------------- Function: AI Response with RAG + Gemini --------------------
def get_gemini_response(prompt):
    """Generates AI response using Google Gemini."""
    if not GEMINI_API_KEY:
        return "⚠️ Error: Missing Gemini API Key."
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text if response else "⚠️ No response generated."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# -------------------- Function: Get Stock Data from Yahoo Finance --------------------
def get_stock_data(ticker):
    """Fetch financial metrics (ROE, market cap, current price) from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Company": info.get("longName", ticker),
            "ROE": info.get("returnOnEquity", "N/A"),
            "Current Price": info.get("currentPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A")
        }
    except Exception as e:
        return {"⚠️ Error": f"Could not retrieve stock data: {str(e)}"}

# -------------------- Function: Get Latest Financial News --------------------
def get_financial_news():
    """Fetch latest business news using News API."""
    if not NEWS_API_KEY:
        return [("⚠️ Error: Missing News API Key", "#")]
    
    url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [(article["title"], article["url"]) for article in articles[:5]]
    except Exception as e:
        return [(f"⚠️ Error: {str(e)}", "#")]

# -------------------- Streamlit UI --------------------
st.title("💰 AI-Powered Finance Chatbot (RAG + AI)")

# User Query Input
user_input = st.text_input("🔍 Ask me anything about finance:")

if user_input:
    # Retrieve relevant finance insights using RAG
    retrieved_info = retrieve_financial_info(user_input)

    # Generate AI response using Gemini (combining RAG + AI)
    full_prompt = f"User Query: {user_input}\n\nRetrieved Financial Insights:\n{retrieved_info}"
    response = get_gemini_response(full_prompt)

    # Display AI response
    st.markdown("### 🤖 AI Response:")
    st.write(response)

    # Stock Insights Section
    if "ROE" in user_input or "best stocks" in user_input:
        stock_symbol = st.text_input("📊 Enter stock ticker (e.g., RELIANCE.NS, AAPL, TSLA):")
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol.upper())
            st.json(stock_data)

    # Financial News Section
    st.markdown("### 📰 Latest Financial News")
    for title, url in get_financial_news():
        st.markdown(f"🔹 [{title}]({url})")

# Run using:
# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success2_chatbot.py"

# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import requests
import faiss
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model # type: ignore
import google.generativeai as genai
import groq
import yfinance as yf
import pdfplumber
import pytesseract
import fitz  # PyMuPDF for text extraction
import plotly.express as px
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import dotenv
from concurrent.futures import ThreadPoolExecutor


from utils.news_scraper import fetch_news
from utils.sentiment_analysis import analyze_sentiment
from utils.risk_prediction import fetch_stock_data, train_arima, train_lstm, predict_risk

# Ensure Python can find the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])
stocks = df["Stock"].unique().tolist()
models = ["LSTM", "ARIMA", "SARIMA"]

# st.title("📈 AI-Powered Stock Prediction & Risk Analysis")


def predict_lstm(stock_name):
    try:
        model = load_model(f"models/lstm_{stock_name}.h5")
        scaler = joblib.load(f"models/scaler_{stock_name}.pkl")
    except FileNotFoundError:
        st.error(f"❌ No trained LSTM model found for {stock_name}")
        return None
    
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    close_prices = stock_data["Close"].values.reshape(-1, 1)
    sentiment_scores = stock_data["Sentiment_Score"].values.reshape(-1, 1)

    last_sequence = np.hstack((close_prices, sentiment_scores))[-lookback:]
    last_sequence_scaled = scaler.transform(last_sequence).reshape(1, lookback, 2)

    predicted_price_scaled = model.predict(last_sequence_scaled)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_price_scaled, 0]])[0][0]
    
    return predicted_price

def predict_arima(stock_name):
    try:
        model = joblib.load(f"models/arima_{stock_name}.pkl")
    except FileNotFoundError:
        st.error(f"❌ No trained ARIMA model found for {stock_name}")
        return None
    
    return model.forecast(steps=1)[0]

def predict_sarima(stock_name):
    try:
        model = joblib.load(f"models/sarima_{stock_name}.pkl")  
    except FileNotFoundError:
        st.error(f"❌ No trained SARIMA model found for {stock_name}")
        return None

    stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    exog_features = stock_data["Sentiment_Score"].values[-1:].reshape(-1, 1)
    forecast = model.forecast(steps=1, exog=exog_features)
    return forecast[0]


# Specify the Tesseract path (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

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

# Function to extract text from PDFs with parallel processing
def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    doc = fitz.open(pdf_path)
    
    def extract_page_text(page):
        return page.get_text("text")
    
    with ThreadPoolExecutor() as executor:
        texts = executor.map(extract_page_text, doc)
    
    extracted_text += "\n".join(texts)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    extracted_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
    return extracted_text

# Function to process uploaded documents with caching
@st.cache_resource
def process_uploaded_documents(uploaded_files):
    global raw_texts
    all_chunks = []
    raw_texts = []
    
    for uploaded_file in uploaded_files:
        temp_path = f"temp_uploaded.{uploaded_file.name.split('.')[-1]}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        extracted_text = extract_text_from_pdf(temp_path)
        raw_texts.append(extracted_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        all_chunks.extend(documents)
    
    if all_chunks:
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore
    return None


import os

@st.cache_resource
def load_vectorstore():
    faiss_index_path = "faiss_index"
    
    # Check if FAISS index exists before loading
    if not os.path.exists(f"{faiss_index_path}/index.faiss"):
        st.warning("⚠️ FAISS index not found. Creating a new index...")
        return None  # Return None so the chatbot can rebuild it when documents are uploaded
    
    return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# Function to retrieve document info
def retrieve_financial_info(query, vectorstore):
    if not vectorstore:
        return ["No document uploaded."]
    query_embedding = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(np.array(query_embedding), k=2)
    retrieved_texts = [doc.page_content for doc in results] if results else []
    if not retrieved_texts:
        retrieved_texts = [text for text in raw_texts if query.lower() in text.lower()]
    return retrieved_texts if retrieved_texts else ["No relevant finance documents found."]

# Function to get responses from AI models
def get_groq_response(prompt):
    response = groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text if response else "No response generated."


# Define Navigation Options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Functionality", ["Home", "Finance Chatbot", "Stock Market Analysis","Stock Risk Prediction", "News Sentiment Analysis", "Predict Price"])

if page == "Home":
    st.title("💸🖥️ Welcome to AI Finance Chatbot 🖥️💸")
    st.write("Select a functionality from the left sidebar to begin.")

elif page == "Finance Chatbot":

# Streamlit UI
 st.title("💰 AI Finance Chatbot (Groq + Gemini)")
 mode = st.radio("Select Mode:", ["General Finance Chat", "Document-Based Analysis"])

 if mode == "General Finance Chat":
    user_query = st.text_input("Ask me anything about finance:")
    if user_query:
        with st.spinner("Fetching responses..."):
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
        st.success("📂 Documents indexed for retrieval!")
    user_input = st.text_input("Ask a question related to uploaded documents:")
    if user_input:
        retrieved_info = retrieve_financial_info(user_input, vectorstore)
        full_prompt = f"Financial query: {user_input}\n\nRetrieved Document Insights:\n{retrieved_info}"
        with st.spinner("Fetching responses..."):
            groq_response = get_groq_response(full_prompt)
            gemini_response = get_gemini_response(full_prompt)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 Groq (LLaMA 3)")
            st.write(groq_response)
        with col2:
            st.subheader("🔍 Gemini (Google)")
            st.write(gemini_response)

elif page == "Stock Market Analysis":
    st.title("📊 Real-Time Stock Market Data")
    ticker = st.selectbox("📌 Select a Stock you want to analyze the risk of:", df["Stock"].unique())
    if ticker:
        stock = yf.Ticker(ticker)
        info = stock.info
        st.write(f"**{info['shortName']}**")
        st.write(f"📈 **Current Price:** Rs.{info.get('currentPrice', 'N/A')}")
    if ticker:
        hist = stock.history(period="1mo")
        fig = px.line(hist, x=hist.index, y="Close", title=f"{ticker} Stock Prices")
        st.plotly_chart(fig)

elif page == "Stock Risk Prediction":
 ticker = st.selectbox("📌 Select a Stock you want to analyze the risk of:", df["Stock"].unique())

 if st.button("Predict Risk"):
    stock_data = fetch_stock_data(ticker)
    arima_forecast = train_arima(stock_data)
    lstm_model, scaler = train_lstm(stock_data)
    predicted_price, risk_score = predict_risk(stock_data, lstm_model, scaler)

    st.write(f"**Predicted Price (LSTM):** {predicted_price}")
    st.write(f"**Risk Score:** {risk_score}")

elif page == "News Sentiment Analysis":
    stock_name = st.selectbox("Select a Stock", stocks)
    lookback = 90
    news = fetch_news(stock_name)
    
    for idx, article in enumerate(news):
        title = article.get("title", "No Title")
        summary = article.get("summary", "No Summary")
        sentiment = article.get("sentiment", "Neutral")
        sentiment_score = article.get("sentiment_score", 0.0)

        # Display formatted news output in Streamlit UI
        st.write(f"📢 **News {idx+1}**")
        st.write(f"**Title:** {title}")
        st.write(f"**Summary:** {summary}")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Sentiment Score:** {sentiment_score:.2f}")
        st.write("---")  # Adds a separator between news articles

elif page == "Predict Price":

    stock_name = st.selectbox("Select a Stock", stocks)
    model_name = st.selectbox("Select a Model", models)
    lookback = 90
    prediction = None
    if model_name == "LSTM":
        prediction = predict_lstm(stock_name)
    elif model_name == "ARIMA":
        prediction = predict_arima(stock_name)
    elif model_name == "SARIMA":
        prediction = predict_sarima(stock_name)
    
    if prediction is not None:
        st.success(f"💰 Predicted Closing Price for {stock_name} using {model_name}: ₹{prediction:.2f}")
    else:
        st.error("⚠️ Prediction failed. Check if the model exists.")



# python -m streamlit run app.py 
# import faiss
# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import sys
# import matplotlib as plt
# import os
# from tensorflow.keras.models import load_model # type: ignore
# import google.generativeai as genai
# import groq
# import yfinance as yf
# import pdfplumber
# import pytesseract
# import fitz  # PyMuPDF for text extraction
# import plotly.express as px
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# import dotenv
# from concurrent.futures import ThreadPoolExecutor
# import fitz
# from langchain.embeddings import OpenAIEmbeddings
# from transformers import T5Tokenizer, T5ForConditionalGeneration


# from utils.news_scraper import fetch_news
# from utils.sentiment_analysis import analyze_sentiment
# from utils.risk_prediction import fetch_stock_data, train_arima, train_lstm, predict_risk


# import streamlit as st
# import pandas as pd
# import joblib
# import os
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import pandas as pd
# import joblib
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import matplotlib.pyplot as plt
# import pandas as pd
# import joblib
# import os
# import streamlit as st
import faiss
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
import groq
import yfinance as yf
import pdfplumber
import pytesseract
import fitz  # PyMuPDF for text extraction
import plotly.express as px
from tensorflow.keras.models import load_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings import OpenAIEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.news_scraper import fetch_news
from utils.sentiment_analysis import analyze_sentiment
from utils.risk_prediction import fetch_stock_data, train_arima, train_lstm, predict_risk

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])
    return df

df = load_data()

# Ensure models directory exists
os.makedirs("models", exist_ok=True)


# Load dataset, stock list , model list
df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])
stocks = df["Stock"].unique().tolist()
models = ["ARIMA", "SARIMA"]

def train_model(stock_name, model_type):
    """
    Trains an ARIMA or SARIMAX model for the given stock.
    """
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    
    if stock_data.empty:
        print(f"❌ No data found for {stock_name}")
        return None, None

    close_prices = stock_data["Close"].values
    exog_features = stock_data["Sentiment_Score"].values if "Sentiment_Score" in stock_data.columns else None

    try:
        if model_type == "ARIMA":
            print(f"📊 Training ARIMA model for {stock_name}...")
            model = ARIMA(close_prices, order=(5, 1, 0))
        
        elif model_type == "SARIMA":
            print(f"📊 Training SARIMAX model for {stock_name}...")
            
            if exog_features is None:
                raise ValueError("SARIMAX requires exogenous features but none were found!")

            model = SARIMAX(close_prices, order=(1,1,1), seasonal_order=(1,1,1,12), exog=exog_features)

        # Fit model
        fitted_model = model.fit()

        # Save model
        model_path = f"models/{model_type.lower()}_{stock_name}.pkl"
        joblib.dump(fitted_model, model_path)

        print(f"✅ Model trained and saved: {model_path}")
        return fitted_model, model_path
    except Exception as e:
        print(f"❌ Error training {model_type} model for {stock_name}: {e}")
        return None, None



# def forecast_stock_price(stock_name, model_type, periods=7):
#     stock_data = df[df["Stock"] == stock_name].sort_values("Date")
#     close_prices = stock_data["Close"]

#     model_path = f"models/{model_type.lower()}_{stock_name}.pkl"

#     if os.path.exists(model_path):
#         model = joblib.load(model_path)
#     else:
#         st.warning(f"⚠️ No {model_type} model found for {stock_name}. Training a new one...")
#         model, _ = train_model(stock_name, model_type)

#     if "Sentiment_Score" in stock_data.columns:
#         last_exog_value = stock_data["Sentiment_Score"].iloc[-1]  # Get last known value
#         future_exog_values = [last_exog_value] * periods  # Repeat for forecast steps
#     else:
#         future_exog_values = None

#     # Forecasting with or without exogenous values
#     if model_type == "SARIMA" and future_exog_values is not None:
#         forecast = model.forecast(steps=periods, exog=future_exog_values)
#     else:
#         forecast = model.forecast(steps=periods)
#     # Forecasting
#     # forecast = model.forecast(steps=periods)

#     # Generate forecast dates
#     last_date = stock_data["Date"].max()
#     forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq="D")[1:]

#     # Limit actual data to past 1 year
#     one_year_ago = last_date - pd.DateOffset(years=1)
#     stock_data_last_year = stock_data[stock_data["Date"] >= one_year_ago]

#     # Plot results
#     plt.figure(figsize=(10, 5))
    
#     # Plot last 1 year of actual stock prices
#     plt.plot(stock_data_last_year["Date"], stock_data_last_year["Close"], label="Actual Prices", color="blue")
    
#     # Plot forecast with visible daily points
#     plt.plot(forecast_dates, forecast, label="Forecast", color="red", linestyle="dashed", marker="o", markersize=5)

#     plt.legend()
#     plt.xlabel("Date")
#     plt.ylabel("Stock Price")
#     plt.title(f"{model_type} Prediction for {stock_name}")

#     st.pyplot(plt)

#     # Display predicted prices in a table
#     forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})
#     forecast_df["Date"] = forecast_df["Date"].dt.strftime('%Y-%m-%d')  # Format date for better readability
    
#     st.subheader(f"📊 Predicted Prices for Next {periods} Days:")
#     st.dataframe(forecast_df.style.format({"Predicted Price": "{:.2f}"}), width=500)


# def forecast_stock_price(stock_name, model_type, periods=7):
#     stock_data = df[df["Stock"] == stock_name].sort_values("Date")
#     close_prices = stock_data["Close"]

#     model_path = f"models/{model_type.lower()}_{stock_name}.pkl"

#     if os.path.exists(model_path):
#         model = joblib.load(model_path)
#     else:
#         st.warning(f"⚠️ No {model_type} model found for {stock_name}. Training a new one...")
#         model, _ = train_model(stock_name, model_type)

#     if "Sentiment_Score" in stock_data.columns:
#         last_exog_value = stock_data["Sentiment_Score"].iloc[-1]  # Get last known value
#         future_exog_values = [last_exog_value] * periods  # Repeat for forecast steps
#     else:
#         future_exog_values = None

#     # Forecasting
#     if model_type == "SARIMA" and future_exog_values is not None:
#         forecast = model.forecast(steps=periods, exog=future_exog_values)
#     else:
#         forecast = model.forecast(steps=periods)

#     # Generate forecast dates
#     last_date = stock_data["Date"].max()
#     forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq="D")[1:]

#     # Prepare data for visualization
#     forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})
#     forecast_df["Date"] = forecast_df["Date"].dt.strftime('%Y-%m-%d')

#     # Limit actual data to past 1 month
#     stock_data_recent = stock_data[stock_data["Date"] >= last_date - pd.DateOffset(months=1)]

#     # Combine actual and forecast data for a single plot
#     stock_data_recent["Type"] = "Actual"
#     forecast_df["Type"] = "Forecast"
#     forecast_df["Close"] = forecast_df["Predicted Price"]

#     combined_df = pd.concat([stock_data_recent, forecast_df], ignore_index=True)

#     # Plot using Plotly
#     fig = px.line(
#         combined_df, 
#         x="Date", 
#         y="Close", 
#         color="Type",
#         title=f"{stock_name} Stock Price Forecast ({model_type})",
#         labels={"Close": "Stock Price", "Date": "Date"},
#         markers=True
#     )
#     fig.update_traces(line=dict(width=2))  # Adjust line thickness for visibility

#     st.plotly_chart(fig, use_container_width=True)

#     # Display predicted prices in a table
#     st.subheader(f"📊 Predicted Prices for Next {periods} Days:")
#     st.dataframe(forecast_df.style.format({"Predicted Price": "{:.2f}"}), width=500)
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def forecast_stock_price(stock_name, model_type, periods=7):
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    close_prices = stock_data["Close"]

    model_path = f"models/{model_type.lower()}_{stock_name}.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.warning(f"⚠️ No {model_type} model found for {stock_name}. Training a new one...")
        model, _ = train_model(stock_name, model_type)

    if "Sentiment_Score" in stock_data.columns:
        last_exog_value = stock_data["Sentiment_Score"].iloc[-1]  # Get last known value
        future_exog_values = [last_exog_value] * periods  # Repeat for forecast steps
    else:
        future_exog_values = None

    # Forecasting
    if model_type == "SARIMA" and future_exog_values is not None:
        forecast = model.forecast(steps=periods, exog=future_exog_values)
    else:
        forecast = model.forecast(steps=periods)

    # Generate forecast dates
    last_date = stock_data["Date"].max()
    forecast_dates = pd.date_range(start=last_date, periods=periods+1, freq="D")[1:]

    # Prepare data for visualization
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})
    forecast_df["Date"] = forecast_df["Date"].dt.strftime('%Y-%m-%d')

    # Limit actual data to past 1 month
    stock_data_recent = stock_data[stock_data["Date"] >= last_date - pd.DateOffset(months=1)]

    # Create a Plotly figure
    fig = go.Figure()

    # Plot actual stock prices
    fig.add_trace(go.Scatter(
        x=stock_data_recent["Date"], 
        y=stock_data_recent["Close"],
        mode='lines+markers',
        marker=dict(color='blue', size=6),  # Blue markers for actual data
        name='Actual Prices'
    ))

    # Plot forecasted stock prices with thinner red markers
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], 
        y=forecast_df["Predicted Price"],
        mode='lines+markers',
        line=dict(color='red', dash='dash'),
        marker=dict(color='red', size=4),  # Thinner red markers
        name='Forecast'
    ))

    # Update layout
    fig.update_layout(
        title=f"{stock_name} Stock Price Forecast ({model_type})",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Legend",
        hovermode="x"
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display predicted prices in a table
    st.subheader(f"📊 Predicted Prices for Next {periods} Days:")
    st.dataframe(forecast_df.style.format({"Predicted Price": "{:.2f}"}), width=500)



# Ensure Python can find the utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))



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

if not GROQ_API_KEY or not GEMINI_API_KEY:
    st.error("⚠️ Missing API Keys. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
groq_client = groq.Client(api_key=GROQ_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None
raw_texts = []

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

def summarize_text(text, max_length=500, min_length=50):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@st.cache_resource
def process_uploaded_documents_for_QA(uploaded_files):
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

@st.cache_resource
def process_uploaded_documents_for_summary(uploaded_files):
    summaries = {}
    for uploaded_file in uploaded_files:
        temp_path = f"temp_uploaded.{uploaded_file.name.split('.')[-1]}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        extracted_text = extract_text_from_pdf(temp_path)
        summarized_text = summarize_text(extracted_text)
        summaries[uploaded_file.name] = summarized_text
    return summaries


@st.cache_resource
def load_vectorstore():
    faiss_index_path = "faiss_index"
    
    if not os.path.exists(f"{faiss_index_path}/index.faiss"):
        st.warning("⚠️ FAISS index not found. Creating a new index...")
        return None 
    
    return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_summarization_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, summarization_model = load_summarization_model()

vectorstore = load_vectorstore()

def retrieve_financial_info(query, vectorstore):
    if not vectorstore:
        return ["No document uploaded."]
    query_embedding = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(np.array(query_embedding), k=2)
    retrieved_texts = [doc.page_content for doc in results] if results else []
    if not retrieved_texts:
        retrieved_texts = [text for text in raw_texts if query.lower() in text.lower()]
    return retrieved_texts if retrieved_texts else ["No relevant finance documents found."]

def get_groq_response(prompt):
    response = groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text if response else "No response generated."


st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Functionality", ["Home", "Finance Chatbot","Document oriented Analyzer","Stock Market Analysis & Risk Prediction", "News Sentiment Analysis & Price Prediction","Price Prediction for next N days"])

if page == "Home":
    st.title("💸FinGenius 🖥️")
    st.subheader("A smart financial AI that can predict, analyze, and answer finance-related queries!")
    st.write("Select a functionality from the left sidebar to begin.")

elif page == "Finance Chatbot":

    st.title("💰 Welcome to AI Finance Chatbot (Groq + Gemini)")

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

elif page == "Document oriented Analyzer":
    st.title("💰 Welcome to AI Finance Doc based chatbot & Summarizer")
    uploaded_files = st.file_uploader("Upload financial documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    mode = st.radio("Select Mode:", ["Question Answering Chatbot", "Document-Based Summarizer"])

    if mode == "Question Answering Chatbot":
      if uploaded_files:
          vectorstore = process_uploaded_documents_for_QA(uploaded_files)
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


    if mode == "Document-Based Summarizer":

       if uploaded_files:
        summaries = process_uploaded_documents_for_summary(uploaded_files)
        st.success("📂 Documents processed and summarized!")

        for file_name, summary in summaries.items():
          st.subheader(f"📜 Summary of {file_name}")
          st.write(summary)

elif page == "Stock Market Analysis & Risk Prediction":
    st.title("📊 Real-Time Stock Market Data & Risk Prediction")
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
    
    #Predict Risk:
        stock_data = fetch_stock_data(ticker)
        arima_forecast = train_arima(stock_data)
        lstm_model, scaler = train_lstm(stock_data)
        predicted_price, risk_score = predict_risk(stock_data, lstm_model, scaler)

    st.write(f"**Risk Score:** {risk_score}")

elif page == "News Sentiment Analysis & Price Prediction":
    st.title("📊 Real-Time Stock News based Sentiment Analysis & Price Prediction")
    stock_name = st.selectbox("Select a Stock", stocks)
    model_name = st.selectbox("Select a Model", models)
    lookback = 90
    news = fetch_news(stock_name)

    if model_name == "ARIMA":
        prediction = predict_arima(stock_name)
    elif model_name == "SARIMA":
        prediction = predict_sarima(stock_name)
    
    if prediction is not None:
        st.success(f"💰 Predicted Closing Price for {stock_name} using {model_name}: ₹{prediction:.2f}")
    else:
        st.error("⚠️ Prediction failed. Check if the model exists.")

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

elif page == "Price Prediction for next N days":
  st.title("📈 Stock Price Prediction with ARIMA & SARIMA for next N days")
  selected_stock = st.selectbox("Select a Stock:", df["Stock"].unique())
  model_choice = st.radio("Select Model Type:", ["ARIMA", "SARIMA"])
  forecast_days = st.slider("Select Forecast Period (Days):", min_value=7, max_value=90, value=30)

  if st.button("Predict"):
    forecast_stock_price(selected_stock, model_choice, forecast_days)
# python -m streamlit run app.py 
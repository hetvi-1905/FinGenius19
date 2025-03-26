# gives the response from bot gemini + groq ( as default without user selection) , without the document uploading part 
import streamlit as st
import google.generativeai as genai
import groq  # Groq API for LLaMA 3
import requests
import os

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

# Initialize Groq API
groq_client = groq.Client(api_key=GROQ_API_KEY)

# Function to get response from Groq (LLaMA 3)
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

# Function to get response from Gemini
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

# User Query Section
user_input = st.text_input("Ask me anything about finance:")

if user_input:
    with st.spinner("Fetching responses..."):
        # Get responses from both AI models
        groq_response = get_groq_response(user_input)
        gemini_response = get_gemini_response(user_input)

    # Display both responses side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Groq (LLaMA 3)")
        st.write(groq_response)

    with col2:
        st.subheader("🔍 Gemini (Google)")
        st.write(gemini_response)
# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success7_chatbot.py"


# 💰 AI Finance Chatbot (Groq + Gemini)
# Ask me anything about finance:

# what is the net profit ratio of suyog telematics in 2024?

# 🤖 Groq (LLaMA 3)
# I apologize, but I couldn't find any information on a company called "Suyog Telematics" and its net profit ratio for 2024. There are a few reasons for this:

# Suyog Telematics may be a private company, and therefore, its financial information may not be publicly available.
# The company may not be listed on any stock exchange, which means its financial statements may not be accessible.
# Even if the company is public, its financial statements for 2024 may not be available yet, as financial reporting usually lags behind the current year.
# If you have more information about Suyog Telematics, such as its country of operation or industry, I may be able to help you find the information you're looking for. Alternatively, you can try searching for the company's website, annual reports, or financial statements on databases like Bloomberg, Thomson Reuters, or Edgar.

# 🔍 Gemini (Google)
# I do not have access to real-time financial data, including the current net profit ratio for Suyog Telematics. Financial information like this changes constantly and is usually only available through official company filings, financial news websites that specifically track that company, or paid financial data services.

# To find the net profit ratio for Suyog Telematics in 2024, you should try the following:

# Check Suyog Telematics' website: Their investor relations section might have annual reports or financial statements.
# Look at major financial news sources: Sites like the Bombay Stock Exchange (BSE) website (if listed), the National Stock Exchange of India (NSE) website (if listed), and reputable financial news outlets may publish their financial results.
# Use a financial data provider: Companies like Bloomberg Terminal or Refinitiv Eikon (subscription required) provide detailed financial information for publicly traded companies.
# Remember that 2024 is still ongoing as of my last knowledge update, so the full-year financial results might not be available yet. You'll likely have to wait until their annual report is released.

# 💰 AI Finance Chatbot (Groq + Gemini)
# Ask me anything about finance:

# what is the meaning of roe in stock market ?

# 🤖 Groq (LLaMA 3)
# In the stock market, ROE stands for Return on Equity. It's a financial ratio that measures a company's profitability from shareholders' equity. In other words, it shows how well a company is using its shareholders' investments to generate earnings.

# ROE is calculated by dividing the company's net income (profit) by its shareholders' equity, and the result is expressed as a percentage.

# The formula to calculate ROE is:

# ROE = Net Income / Shareholders' Equity

# This means that for every dollar of shareholders' equity, the company generated a 20% return in profit.

# ROE is an important metric for investors and analysts because it helps them:

# Evaluate a company's profitability: A higher ROE indicates that a company is generating more profit from its shareholders' equity.
# Compare companies: ROE allows investors to compare the profitability of different companies within the same industry or sector.
# Assess management performance: A company's ROE can reflect the effectiveness of its management in generating returns for shareholders.
# Identify potential investment opportunities: A high ROE may indicate a company with strong growth potential, making it a more attractive investment option.
# In general, a higher ROE is considered better, as it indicates that a company is using its shareholders' equity more efficiently to generate profits. However, the acceptable ROE varies across industries and sectors, and a company's ROE should be evaluated in the context of its industry and peer group.

# 🔍 Gemini (Google)
# ROE stands for Return on Equity. In the stock market, it's a financial ratio that measures a company's profitability in relation to its shareholders' equity. Essentially, it shows how effectively a company is using its shareholders' investments to generate profit.

# A higher ROE generally indicates better management and more efficient use of shareholder capital, making the stock potentially more attractive. However, ROE should be considered in context with other financial metrics and industry benchmarks, as a high ROE doesn't automatically guarantee future success. There can be several reasons for high or low ROE, some of which may not be sustainable.'
# '
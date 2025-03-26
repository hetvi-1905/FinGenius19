# groq + gemini - user will be given option to get response from , without the document uploading part 
import streamlit as st
import google.generativeai as genai
import requests
import os
import groq  # Groq API for LLaMA 3
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# Initialize FAISS Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None  # Initialize FAISS storage

# Function to get response from Groq (LLaMA 3)
def get_groq_response(prompt):
    """Fetches AI-generated response from Groq API."""
    if not GROQ_API_KEY:
        return "⚠️ Error: Missing Groq API Key"

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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
        return f"⚠️ Error: {str(e)}"

# Streamlit UI
st.title("💰 AI Finance Chatbot (Groq + Gemini)")

# Let user choose the AI model
ai_model = st.radio("Choose AI Model:", ["Gemini (Google)", "Groq (LLaMA 3)"])

# User Query Section
user_input = st.text_input("Ask me anything about finance:")

if user_input:
    if ai_model == "Gemini (Google)":
        response = get_gemini_response(user_input)
    else:
        response = get_groq_response(user_input)

    st.write("### 🤖 AI Response:")
    st.write(response)

# python -m streamlit run "c:/Users/Admin/OneDrive/Desktop/TSF Project/chatbot/success6_chatbot.py"


## Question : Should I invest in RELIANCE?

# gemini response:
# 🤖 AI Response:
# Whether or not you should invest in Reliance Industries (RIL) depends entirely on your individual circumstances, risk tolerance, and investment goals. There's no simple yes or no answer.

# Arguments for investing in Reliance:

# Diversified Business: RIL has a presence in multiple sectors, including petrochemicals, refining, telecom (Jio), retail (Reliance Retail), and digital services. This diversification can help mitigate risk.
# Strong Financials: Generally, RIL demonstrates strong financial performance with high revenue and profitability.
# Growth Potential: The company is actively expanding into new areas and technologies, suggesting future growth opportunities. Jio's expansion in the telecom sector and Reliance Retail's growth in the Indian retail market are key drivers.
# Market Leader: RIL holds a significant market share in several of its operating sectors.
# Brand Recognition: Reliance is a well-known and established brand in India.
# Arguments against investing in Reliance:

# High Valuation: RIL's stock price can be considered expensive relative to some other companies, depending on the market conditions and valuation metrics used.
# Cyclicality: Some of RIL's businesses, like petrochemicals, are susceptible to commodity price fluctuations, which can impact profitability.
# Regulatory Risks: The company operates in a heavily regulated environment, and changes in regulations could affect its operations.
# Geopolitical Risks: Global events can impact the company's performance, particularly in the energy sector.
# Competition: RIL faces intense competition in various sectors.
# Before making a decision, consider:

# Your risk tolerance: Are you comfortable with the potential for volatility in the stock price?
# Your investment horizon: How long do you plan to hold the investment? Long-term investments generally have a better chance of weathering short-term market fluctuations.
# Your investment goals: What are you hoping to achieve with this investment (e.g., capital appreciation, income generation)?
# Diversification: Don't put all your eggs in one basket. RIL should be part of a diversified portfolio.
# Conduct thorough research: Read financial news, analyst reports, and company filings to gain a comprehensive understanding of RIL's performance and future prospects. Consider consulting a qualified financial advisor.
# In short: RIL can be a valuable addition to a well-diversified portfolio, but it's crucial to conduct your own thorough research and consider your individual circumstances before making any investment decisions. This information is not financial advice.

# Groq AI response:
# 🤖 AI Response:
# Reliance Industries Ltd (RIL) is a large-cap stock and a leader in the Indian energy and materials sector. As an investor, it's essential to evaluate the company's fundamentals, growth prospects, and industry trends before making a decision. Here's a brief analysis to help you make an informed decision:

# Reasons to consider investing in Reliance:

# Diversified business portfolio: RIL has a diversified business portfolio, including energy (refining, petrochemicals, and exploration & production), retail (Reliance Retail), telecommunications (Jio), and media (Network18).
# Strong brand portfolio: RIL has a strong brand portfolio, including Jio, which has disrupted the Indian telecommunications industry, and Reliance Retail, which is one of India's largest retail chains.
# Growth potential: RIL is investing heavily in new-age businesses, such as e-commerce, digital payments, and artificial intelligence, which could drive long-term growth.
# Strong financials: RIL has a strong balance sheet, with a net debt-to-equity ratio of 0.15, and a robust cash flow generation ability.
# Dividend yield: RIL has a consistent dividend payment history, with a current dividend yield of around 0.4%.
# Challenges and concerns:

# Valuation: RIL's valuations are relatively high, with a price-to-earnings (P/E) ratio of around 24.5, which is higher than its 5-year average.
# Competition: Intense competition in the energy and retail sectors could impact RIL's market share and profitability.
# Regulatory risks: Changes in government policies and regulations could affect RIL's business, particularly in the energy and telecommunications sectors.
# Global economic uncertainty: RIL's businesses are exposed to global economic trends, and any slowdown could impact its performance.
# What to do:

# Research and analysis: Conduct thorough research and analysis on RIL's business, industry, and financials to form your own opinion.
# Long-term perspective: Consider investing in RIL with a long-term perspective (3-5 years or more), as the company's growth prospects are likely to unfold over time.
# Diversification: Ensure that RIL is a part of a diversified portfolio, with adequate exposure to other sectors and asset classes.
# Valuation: Keep an eye on RIL's valuations and consider investing if the stock price falls to more attractive levels.
# Consult a financial advisor: If you're new to investing or unsure about RIL's prospects, consider consulting a financial advisor or investment professional.
# Ultimately, the decision to invest in Reliance Industries Ltd depends on your individual financial goals, risk tolerance, and investment horizons. It's essential to carefully evaluate the company's prospects and risks before making a decision.

# 💰 AI Finance Chatbot (Groq + Gemini)
# Choose AI Model:


# Gemini (Google) --> selected


# Groq (LLaMA 3)

# Ask me anything about finance:

### What is the net profit ratio of Suyog in 2024?
# 🤖 AI Response:
# I need more information to answer this question. The net profit ratio requires knowing Suyog's net profit and 
# revenue (or sales) for 2024. The formula is:

# Net Profit Ratio = (Net Profit / Revenue) * 100

# Please provide Suyog's net profit and revenue figures for 2024.

# 💰 AI Finance Chatbot (Groq + Gemini)
# Choose AI Model:


# Gemini (Google) 


# Groq (LLaMA 3) --> selected

# Ask me anything about finance:

### What is the net profit ratio of Suyog in 2024?
# 🤖 AI Response:
# I'm happy to help! However, I don't have enough information to provide the net profit ratio of Suyog in 2024. 
# Can you please provide more context or details about Suyog, such as its industry, revenue, and expenses? That way, 
# I can assist you in calculating the net profit ratio.
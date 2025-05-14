# **FinGenius: AI-Powered Financial Insights & Prediction Platform**

## **📌 Overview**
Welcome to **FinGenius**, an advanced AI-powered financial analysis and prediction platform designed to provide insights into stock markets, document-based financial intelligence, and AI-driven chatbot assistance. This interactive platform comprises **five distinct modules**, each catering to a unique aspect of financial analytics and forecasting.

## 💻Technologies used:
- Time Series Forecasting
- NLP
- LangChain
- Gen AI 
- Large Language Models
---
## **🛠 Project Structure**  

📂 **Folder Organization**  

```
FinGenius/
│── chatbot/               # AI chatbot using Groq LLaMA3-8B & Gemini API
│── data/                  # Financial & stock market dataset storage
│── data_collection/       # Scripts for real-time data fetching (Yahoo Finance, Google Trends)
│── faiss_index/           # FAISS vector database for document retrieval
│── fintech/               # Virtual Env
│── models/                # Pretrained ML models for stock forecasting & risk assessment
│── utils/                 # Helper functions for data processing & API integration
│── app.py                 # Streamlit web application
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

---

## **🛠️ Modules & Features**

### **1️⃣ Home 🏠**
- Displays the project name **"FinGenius"** prominently.
- Provides a concise one-line description of the platform’s purpose.

---

### **2️⃣ Finance Chatbot 🤖**
- AI-powered chatbot specialized in **finance-related queries**.
- Utilizes **Groq’s LLaMA3-8B-8192** and **Gemini-1.5-Flash** models via API keys.
- Functions similarly to ChatGPT but with a **finance-specific knowledge base**.
- Capable of answering **general financial questions, market trends, and investment insights**.

---

### **3️⃣ Document-Oriented Analyzer 📄**
A **dual-functional document analysis module**, allowing users to interact with uploaded financial documents:

#### **📌 3.1 - QA-Based Document Querying**
- Allows users to upload financial documents.
- Uses **FAISS RecursiveCharacterTextSplitter** and **Document Library** to **index & process** documents.
- Enables users to **ask finance-specific questions** about the uploaded document.
- Retrieves relevant text segments and utilizes **Groq & Gemini models** for AI-powered answers.

#### **📌 3.2 - Document Summarization**
- Uses **"summarize_text"** function to process the uploaded document.
- AI-generated concise summary using **Groq & Gemini**.
- Extracts key insights, enabling users to **grasp financial reports quickly**.

---

### **4️⃣ Stock Market Analysis & Risk Prediction 📈**
- Users select a stock from the **dropdown menu**.
- Retrieves **real-time stock price data**.
- Plots **closing prices over the past month**.
- Generates a **Risk Sentiment Score** (Low, Medium, or High) based on stock market trends.
- Helps investors assess the **risk level of a stock before investing**.

---

### **5️⃣ News Sentiment Analysis & Stock Price Prediction 📰📊**
A **hybrid module** that combines **news sentiment analysis** with **stock price forecasting**:

🔹 **User Inputs:**
  - Select **Stock Name** from dropdown.
  - Choose **Forecasting Model** (**ARIMA or SARIMA**).

🔹 **Outputs:**

1️⃣ **Next-Day Stock Price Prediction** using **ARIMA or SARIMAX + Top 3 Latest News Sentiments related to that Ticker stock**.
   - Implements **Auto ARIMA** to determine optimized **p, q, d** values for better accuracy.

2️⃣ **Top 3 Financial News Articles & Sentiment Analysis**
   - Extracts latest news using **Yahoo Finance API**.
   - Applies **ProsusAI/FinBERT** model (Finance-specific Transformer) to determine:
     - **News Sentiment** (Positive, Neutral, Negative).
     - **Sentiment Score** (Confidence level in prediction).

---

### **6️⃣ Price Prediction for Next N Days 🔮**
- **User Inputs:**
  - Select **Stock Name**.
  - Choose **Forecasting Model** (**ARIMA or SARIMA**).
  - Adjust the slider for **"N" (Number of Days)** to predict future prices.

- **Outputs:**
  - **Predicted stock prices for the next N days**.
  - **Line plot showcasing future trend predictions** alongside past price trends.
  - Helps traders and investors **forecast market movements** with greater accuracy.

---

## **🔥 Features**
✅ **AI-Driven Finance Chatbot** with **Groq LLaMA3-8B-8192** & **Gemini-1.5-Flash**.  
✅ **Real-Time Market Analysis & Stock Forecasting** using ARIMA & SARIMAX  📈.
✅ **Document-Based Financial Insights** with **FAISS** & Summarization.  
✅ **Sentiment Analysis on Financial News** with FinBERT Transformer.  
✅ **Interactive, User-Friendly Interface** powered by Streamlit.  

---

## **🚀 Tech Stack & Integrations**
💻 **Programming Language:** Python 
📡 **APIs:** Yahoo Finance, Google Trends, Groq, Gemini  
🧠 **Machine Learning Models:** ARIMA, SARIMA, Auto-ARIMA, LLaMA3-8B, FinBERT, Langchiain  
📊 **Visualization:** Matplotlib, Plotly, Seaborn  
💾 **Database:** FAISS for document-based AI retrieval  
⚙️ **Frameworks:** Streamlit, Flask 
☁️ **Deployment:** Docker, AWS  ( optional ) 

---


## **🚀 How to Run the Project**

### **1️⃣ Clone the Repository**
```bash
$ git clone https://github.com/your-repo/FinGenius.git
$ cd FinGenius
```

### **2️⃣ Install Dependencies**
```bash
$ pip install -r requirements.txt
```

### **3️⃣ Run Streamlit App**
```bash
$ streamlit run app.py
or
$ python -m streamlit run app.py
```
- Tip: can also make a virtual environment in the folder where this project is like the one in mine named "fintech" and then activate it and then go to step 2 above 
- Imp Note: Its important to download tesseract.exe from https://github.com/UB-Mannheim/tesseract/wiki and put in the folder and copy that absolute folder path in the place of "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" in appy.py file to not get any error while extracting the information from the uploaded documents
- And lastly, dont forget to change the API keys in the chatbot/env file to get the code running 

---
## **📌 Usage Guide**  
1️⃣ Open **Streamlit UI** in your browser.  
2️⃣ **Upload** financial reports & analyze insights.  
3️⃣ **Ask Chatbot** financial queries using Groq/Gemini AI.  
4️⃣ **Predict stock prices** & assess market trends.  

---

## **📊 Models Used**  
🔹 **LLaMA3-8B-8192 (Groq API)** – AI chatbot for financial queries  
🔹 **Gemini-1.5-Flash** – Advanced chatbot capabilities  
🔹 **FAISS** – Financial document retrieval & analysis  
🔹 **ARIMA & SARIMAX** – Time-series forecasting for stock prices  
🔹 **FinBERT** – Sentiment analysis for financial news  

---

## **🔗 APIs & Integrations**  
🌍 **Yahoo Finance API** – Fetches real-time stock data  
📈 **Google Trends API** – Market trend analysis  
🤖 **Grok API** – AI chatbot integration  
📂 **FAISS Indexing** – Intelligent document search  

---

## **UI Images** 
 

<img src="https://github.com/user-attachments/assets/47bbed29-cda9-454e-8e62-05bb40172f63" width="500" height="300">

<img src="https://github.com/user-attachments/assets/bce1d9b5-9fd2-415f-8c51-223e2d53381c" width="500" height="300">

<img src="https://github.com/user-attachments/assets/45b23a2e-9f0a-469f-961a-49188360de7e" width="500" height="300">

<img src="https://github.com/user-attachments/assets/fb98521c-13ad-4a43-94a9-e39801822398" width="500" height="300">

<img src="https://github.com/user-attachments/assets/8fbb5909-9a38-4700-adbc-6a5cf512e126" width="500" height="300">

<img src="https://github.com/user-attachments/assets/b37ff26d-de86-4d31-be1a-28b85e02824a" width="500" height="300">

<img src="https://github.com/user-attachments/assets/28896442-3569-4d1e-952d-e6059722b351" width="500" height="300">

<img src="https://github.com/user-attachments/assets/21fb641e-def4-41f1-9f59-1268072ed8d0" width="500" height="300">

<img src="https://github.com/user-attachments/assets/96e58f2e-3c54-44ac-a7f6-6575dfeb0104" width="500" height="300">

---


## **📩 Contact**  
📧 **Email:** hetvibhora192@gmail.com  
🌍 **GitHub:** [Hetvi Bhora](https://github.com/hetvi-1905)  

📢 **Join in revolutionizing financial analysis with AI! 🚀**

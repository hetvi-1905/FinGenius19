# FinTech19

# **📌 FinGenius: AI-Powered Financial Analysis & Prediction Platform**  

## **📖 Overview**  
🚀 **FinGenius** is an AI-powered financial analysis and prediction platform designed to provide insights into financial documents, stock trends, and investment risks. It utilizes advanced AI models to deliver **accurate predictions, sentiment analysis, and document-based intelligence**.  

## **🛠 Project Structure & Modules**  

📂 **Folder Organization**  

FinGenius/
│── chatbot/               # AI chatbot using Groq LLaMA3-8B & Gemini API
│── data/                  # Financial & stock market dataset storage
│── data_collection/       # Scripts for real-time data fetching (Yahoo Finance, Google Trends)
│── faiss_index/           # FAISS vector database for document retrieval
│── fintech/               # Core financial models for prediction & analysis
│── models/                # Pretrained ML models for stock forecasting & risk assessment
│── utils/                 # Helper functions for data processing & API integration
│── app.py                 # Streamlit web application
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

--- 

### **📌 Module Breakdown**  

### **1️⃣ AI Finance Chatbot (🤖 chatbot/)**
- Implements **Groq's LLaMA3-8B-8192** & **Gemini-1.5-Flash** for answering finance-related queries.  
- Capable of analyzing financial documents & summarizing insights.  
- Integrated with **FAISS** for vector-based document retrieval.  

### **2️⃣ Financial Document Analysis (📂 faiss_index/)**
- Allows users to **upload financial PDFs** (e.g., balance sheets, annual reports).  
- Uses **FAISS vector search** to retrieve relevant insights.  
- AI-based NLP processing to extract **key financial data**.  

### **3️⃣ Stock Price Prediction (📊 fintech/)**
- Implements **ARIMA, SARIMAX, LSTM** models for time-series forecasting.  
- Fetches **real-time stock data** from **Yahoo Finance API**.  
- Predicts short-term **price movements & trends** for investment decisions.  

### **4️⃣ Market Sentiment Analysis (📈 models/)**
- Uses **FinBERT (Finance-specific BERT model)** for **sentiment analysis**.  
- Fetches **real-time financial news** from **Google Trends & Bloomberg**.  
- Computes a **sentiment score** to assess stock investment risk.  

### **5️⃣ Real-Time Data Collection (📡 data_collection/)**
- Integrates **Yahoo Finance API** for live stock prices.  
- Scrapes **Google Trends** for market sentiment analysis.  
- Stores data in structured format for **faster ML processing**.  

### **6️⃣ Web Application (🖥️ app.py)**
- **Streamlit UI** for interacting with financial data & AI chatbot.  
- Users can **upload reports, analyze stocks, and chat with AI**.  
- Provides **visual graphs & insights** on financial trends.  

## **🔥 Features**  
✅ **AI Finance Chatbot** (Powered by **Groq LLaMA3-8B-8192** & **Gemini-1.5-Flash**)  
✅ **Financial Document Analysis** (PDF Upload & Retrieval using **FAISS**)  
✅ **Stock Price Prediction** (ARIMA, SARIMAX, LSTM) 📈  
✅ **Market Sentiment Analysis** (FinBERT for investment risk assessment)  
✅ **Real-time Data Fetching** (Yahoo Finance, Google Trends)  

## **🚀 Tech Stack**  
💻 **Programming Language:** Python  
⚙️ **Frameworks:** Streamlit, Flask  
🧠 **Machine Learning Models:** ARIMA, SARIMAX, LSTM, FinBERT  
📡 **APIs:** Yahoo Finance, Google Trends, Grok  
🗂 **Database:** FAISS for document search  
☁️ **Deployment:** Docker, AWS  

## **📥 Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-repo/FinGenius.git
cd FinGenius
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Application**  
```bash
streamlit run app.py
```

## **📌 Usage Guide**  
1️⃣ Open **Streamlit UI** in your browser.  
2️⃣ **Upload** financial reports & analyze insights.  
3️⃣ **Ask Chatbot** financial queries using Groq/Gemini AI.  
4️⃣ **Predict stock prices** & assess market trends.  

## **📊 Models Used**  
🔹 **LLaMA3-8B-8192 (Groq API)** – AI chatbot for financial queries  
🔹 **Gemini-1.5-Flash** – Advanced chatbot capabilities  
🔹 **FAISS** – Financial document retrieval & analysis  
🔹 **ARIMA & SARIMAX** – Time-series forecasting for stock prices  
🔹 **FinBERT** – Sentiment analysis for financial news  

## **🔗 APIs & Integrations**  
🌍 **Yahoo Finance API** – Fetches real-time stock data  
📈 **Google Trends API** – Market trend analysis  
🤖 **Grok API** – AI chatbot integration  
📂 **FAISS Indexing** – Intelligent document search  

## **🤝 Contributing**  
👥 **Want to contribute?** Feel free to open an issue or submit a pull request. Let's build something amazing together! 🚀  

---

## **📩 Contact**  
📧 **Email:** your-email@example.com  
🌍 **GitHub:** [your-repo](https://github.com/your-repo/FinGenius)  


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

Now, this **README.md** is **fully structured, includes module-wise details, and is visually appealing**! 🚀 Let me know if you need any more refinements. 😊
















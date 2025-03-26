
import requests
import pandas as pd
import feedparser
from yahoo_fin import news as yf_news
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_nifty50_stocks():
    return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "WAAREEENER.NS"]

def fetch_yahoo_news(stock):
    try:
        news_articles = yf_news.get_yf_rss(stock)
        headlines = [article['title'] for article in news_articles[:10]]                                                                    
        return headlines
    except Exception as e:
        print(f"Error fetching Yahoo Finance news for {stock}: {e}")
        return []

def fetch_google_news(stock):
    try:
        feed_url = f"https://news.google.com/rss/search?q={stock}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(feed_url)
        headlines = [entry.title for entry in feed.entries[:10]]
        return headlines
    except Exception as e:
        print(f"Error fetching Google News for {stock}: {e}")
        return []

def analyze_sentiment(headlines):
    scores = [sia.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores) if scores else 0

if __name__ == "__main__":
    stocks = get_nifty50_stocks()
    sentiment_data = []

    for stock in stocks:
        yahoo_news = fetch_yahoo_news(stock)
        google_news = fetch_google_news(stock)
        all_news = yahoo_news + google_news
        sentiment_score = analyze_sentiment(all_news)

        # ✅ Load stock data for the stock to get 'Date' column
        stock_df = pd.read_csv(f"data/{stock}.csv")

        # ✅ Assign the same sentiment score to each date
        stock_df["Stock"] = stock
        stock_df["Sentiment_Score"] = sentiment_score

        sentiment_data.append(stock_df[["Date", "Stock", "Sentiment_Score"]])

    # ✅ Merge all stock sentiment data
    final_sentiment_df = pd.concat(sentiment_data)
    final_sentiment_df.to_csv("data/news_sentiment.csv", index=False)
    
    print("✅ Saved news_sentiment.csv with Date column")

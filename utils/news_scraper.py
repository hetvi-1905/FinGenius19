import yfinance as yf
from utils.sentiment_analysis import analyze_sentiment

news_data = []

def fetch_news(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    news = stock.news  # Fetches recent news articles
    news_data = []
    # Get only the top 3 latest news articles
    latest_news = news[:3]  

    for idx, news_item in enumerate(latest_news):
        content = news_item.get("content", {})
        title = content.get("title", "No Title")
        summary = content.get("summary", "No Summary")

        # Perform Sentiment Analysis
        sentiment_label, sentiment_scores = analyze_sentiment(summary)

        # Extract the highest sentiment category
        highest_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        highest_score = sentiment_scores[highest_sentiment]
        
        # news_data = []
        # Store result
        news_data.append({
            "title": title,
            "summary": summary,
            "sentiment": highest_sentiment,  
            "sentiment_score": highest_score, 
        })

    return news_data

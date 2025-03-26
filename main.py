from utils.news_scraper import fetch_stock_news, analyze_sentiment 

def get_stock_sentiment(stock):
    news = fetch_stock_news(stock)
    sentiment_score = analyze_sentiment(news)
    return sentiment_score, news

if __name__ == "__main__":
    stock = "TCS.NS"
    sentiment_score, news = get_stock_sentiment(stock)

    print(f"News Sentiment Score for {stock}: {sentiment_score:.2f}")
    print("Top Headlines:")
    for n in news:
        print("-", n)

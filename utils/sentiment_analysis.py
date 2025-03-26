# from textblob import TextBlob

# def analyze_sentiment(news_headlines):
#     """
#     Analyzes sentiment of a list of news headlines.
#     Returns a numerical sentiment score.
#     """
#     if not news_headlines:
#         return 0  # Default neutral sentiment if no news is available

#     sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in news_headlines]
#     avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score

#     return round(avg_sentiment, 2)  # Return a rounded sentiment score

# from textblob import TextBlob

# def analyze_sentiment(text):
#     analysis = TextBlob(text)
#     polarity = analysis.sentiment.polarity
#     sentiment_label = 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
#     return polarity, sentiment_label
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# # Load FinBERT model
# model_name = "yiyanghkust/finbert-tone"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# def analyze_sentiment(text):
#     """Analyze sentiment of the given text and return both category and numeric scores."""
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
    
#     scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
#     labels = ["negative", "neutral", "positive"]
    
#     sentiment_scores = dict(zip(labels, scores))  # Convert to dictionary
    
#     # Get sentiment category
#     sentiment_category = max(sentiment_scores, key=sentiment_scores.get)  
    
#     return sentiment_category, sentiment_scores  # Returns both label and scores
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load FinBERT model
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    """Analyze sentiment of the given text and return both category and numeric scores."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
    labels = ["negative", "neutral", "positive"]
    
    sentiment_scores = dict(zip(labels, scores))  # Convert to dictionary
    
    # Get sentiment category
    sentiment_category = max(sentiment_scores, key=sentiment_scores.get)  
    
    return sentiment_category, sentiment_scores  # Returns both label and scores


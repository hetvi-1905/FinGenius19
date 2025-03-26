import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima  # Auto SARIMAX for best (p, d, q) & (P, D, Q, s)

# Load dataset
df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])

def find_best_sarimax_order(series, exog_series):
    """
    Automatically finds the best (p, d, q) and (P, D, Q, s) using Auto ARIMA.
    """
    print("🔍 Finding optimal (p, d, q) and (P, D, Q, s) values using Auto ARIMA...")
    
    auto_model = auto_arima(series, exogenous=exog_series, seasonal=True, m=5,
                            trace=True, suppress_warnings=True)
    
    best_order = auto_model.order
    best_seasonal_order = auto_model.seasonal_order
    print(f"✅ Best SARIMAX Order Found: {best_order}, Seasonal: {best_seasonal_order}")
    
    return best_order, best_seasonal_order

def train_sarimax(stock_name):
    """
    Trains a SARIMAX model with the best (p, d, q) and (P, D, Q, s) and saves it.
    """
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    
    if stock_data.empty:
        print(f"❌ No data found for {stock_name}")
        return None

    close_prices = stock_data["Close"].values
    sentiment_scores = stock_data["Sentiment_Score"].values  # Exogenous variable

    try:
        # Find best SARIMAX order
        best_order, best_seasonal_order = find_best_sarimax_order(close_prices, sentiment_scores)
        
        # Train SARIMAX model
        model = SARIMAX(close_prices, exog=sentiment_scores, order=best_order, seasonal_order=best_seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Save the trained model
        joblib.dump(fitted_model, f"models/sarima_{stock_name}.pkl")
        print(f"✅ SARIMAX model trained and saved for {stock_name} (Order: {best_order}, Seasonal: {best_seasonal_order})")
        
        return fitted_model

    except Exception as e:
        print(f"❌ Error training SARIMAX model for {stock_name}: {e}")
        return None

def predict_next_day_price_sarimax(stock_name):
    """
    Loads the trained SARIMAX model and predicts the next day's closing price.
    """
    try:
        model = joblib.load(f"models/sarima_{stock_name}.pkl")
    except Exception:
        print(f"⚠️ No saved model found for {stock_name}. Training a new one...")
        model = train_sarimax(stock_name)
    
    if model is None:
        print(f"❌ Could not load or train SARIMAX model for {stock_name}")
        return None

    stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    sentiment_scores = stock_data["Sentiment_Score"].values  # Exogenous variable

    # Forecast the next day's stock price
    forecast = model.forecast(steps=1, exog=[sentiment_scores[-1]])[0]
    print(f"📈 Predicted closing price for {stock_name} (next day): ₹{forecast:.2f}")
    
    return forecast

# Example usage:
stock_name = "WAAREEENER.NS"  # Change this to any stock in your dataset
predict_next_day_price_sarimax(stock_name)

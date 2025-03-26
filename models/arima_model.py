# import pandas as pd
# import joblib
# from statsmodels.tsa.arima.model import ARIMA

# # Load the dataset
# df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])

# def train_arima(stock_name, order=(5,1,0)):
#     """
#     Trains an ARIMA model on the closing prices of a given stock and saves it.
#     """
#     stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    
#     if stock_data.empty:
#         print(f"❌ No data found for {stock_name}")
#         return None

#     close_prices = stock_data["Close"].values
    
#     try:
#         model = ARIMA(close_prices, order=order)
#         fitted_model = model.fit()
        
#         # Save the trained model
#         joblib.dump(fitted_model, f"models/arima_{stock_name}.pkl")
#         print(f"✅ ARIMA model trained and saved for {stock_name}")
#         return fitted_model

#     except Exception as e:
#         print(f"❌ Error training ARIMA model for {stock_name}: {e}")
#         return None

# def predict_next_day_price(stock_name):
#     """
#     Loads the trained ARIMA model and predicts the next day's closing price.
#     """
#     try:
#         model = joblib.load(f"models/arima_{stock_name}.pkl")
#     except Exception:
#         print(f"⚠️ No saved model found for {stock_name}. Training a new one...")
#         model = train_arima(stock_name)
    
#     if model is None:
#         print(f"❌ Could not load or train ARIMA model for {stock_name}")
#         return None

#     # Forecast the next day's stock price
#     forecast = model.forecast(steps=1)[0]
#     print(f"📈 Predicted closing price for {stock_name} (next day): ₹{forecast:.2f}")
#     return forecast

# # Example usage:
# stock_name = "RELIANCE.NS"  # Change this to any stock in your dataset
# predict_next_day_price(stock_name)
import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # Auto ARIMA for best (p, d, q)

# Load dataset
df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])

def find_best_arima_order(series):
    """
    Automatically finds the best (p, d, q) order using Auto ARIMA.
    """
    print("🔍 Finding optimal (p, d, q) values using Auto ARIMA...")
    
    # Run Auto ARIMA to find best order
    auto_model = auto_arima(series, seasonal=False, trace=True, suppress_warnings=True)
    
    best_order = auto_model.order
    print(f"✅ Best ARIMA Order Found: {best_order}")
    
    return best_order

def train_arima(stock_name):
    """
    Trains an ARIMA model with the best (p, d, q) order and saves it.
    """
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")
    
    if stock_data.empty:
        print(f"❌ No data found for {stock_name}")
        return None

    close_prices = stock_data["Close"].values
    
    try:
        # Find best (p, d, q)
        best_order = find_best_arima_order(close_prices)
        
        # Train ARIMA model
        model = ARIMA(close_prices, order=best_order)
        fitted_model = model.fit()
        
        # Save the trained model
        joblib.dump(fitted_model, f"models/arima_{stock_name}.pkl")
        print(f"✅ ARIMA model trained and saved for {stock_name} (Order: {best_order})")
        
        return fitted_model

    except Exception as e:
        print(f"❌ Error training ARIMA model for {stock_name}: {e}")
        return None

def predict_next_day_price(stock_name):
    """
    Loads the trained ARIMA model and predicts the next day's closing price.
    """
    try:
        model = joblib.load(f"models/arima_{stock_name}.pkl")
    except Exception:
        print(f"⚠️ No saved model found for {stock_name}. Training a new one...")
        model = train_arima(stock_name)
    
    if model is None:
        print(f"❌ Could not load or train ARIMA model for {stock_name}")
        return None

    # Forecast the next day's stock price
    forecast = model.forecast(steps=1)[0]
    print(f"📈 Predicted closing price for {stock_name} (next day): ₹{forecast:.2f}")
    
    return forecast

# Example usage:
stock_name = "WAAREEENER.NS"  # Change this to any stock in your dataset
predict_next_day_price(stock_name)

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])

# Function to evaluate saved ARIMA model
def evaluate_saved_arima(stock_name):
    """
    Evaluates a previously trained ARIMA model by comparing its predictions with actual values.
    """
    try:
        # Load the trained ARIMA model
        model = joblib.load(f"models/arima_{stock_name}.pkl")
        print(f"✅ Loaded ARIMA model for {stock_name}")
    except FileNotFoundError:
        print(f"❌ No saved ARIMA model found for {stock_name}. Train it first.")
        return None

    # Get stock data
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")

    if stock_data.shape[0] < 50:
        print(f"⚠️ Not enough data for {stock_name}. Skipping evaluation...")
        return None

    stock_data.set_index("Date", inplace=True)

    # Extract Close Prices
    prices = stock_data["Close"]

    # Split into training (80%) and test (20%)
    train_size = int(len(prices) * 0.8)
    train, test = prices[:train_size], prices[train_size:]

    # Predict next-day price for each test instance
    predictions = []
    history = list(train)  # Start with training data

    for actual_price in test:
        next_pred = model.forecast(steps=1)[0]  # Predict next day
        predictions.append(next_pred)

        # Update history with actual price (for rolling forecast)
        history.append(actual_price)

    # Compute evaluation metrics
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    r2 = r2_score(test, predictions)

    print(f"📊 Evaluation of Saved ARIMA Model for {stock_name}:")
    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📈 MAE: {mae:.2f}")
    print(f"📊 MAPE: {mape:.2f}%")
    print(f"📏 R² Score: {r2:.4f}")
    
    return mae, rmse, mape, r2

# Example Usage:
stock_name = "WAAREEENER.NS"  # Change as needed
evaluate_saved_arima(stock_name)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])

def evaluate_saved_lstm(stock_name, lookback=90):
    """
    Evaluates a previously trained LSTM model by comparing its predictions with actual values.
    """
    try:
        # Load the trained LSTM model
        model = tf.keras.models.load_model(f"models/lstm_{stock_name}.h5", compile=False)
        scaler = joblib.load(f"models/scaler_{stock_name}.pkl")
        print(f"✅ Loaded LSTM model for {stock_name}")
    except FileNotFoundError:
        print(f"❌ No saved LSTM model found for {stock_name}. Train it first.")
        return None

    # Get stock data
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")

    if stock_data.shape[0] < lookback + 10:
        print(f"⚠️ Not enough data for {stock_name}. Skipping evaluation...")
        return None

    # Extract features
    close_prices = stock_data["Close"].values.reshape(-1, 1)
    sentiment_scores = stock_data["Sentiment_Score"].values.reshape(-1, 1)
    
    features = np.hstack((close_prices, sentiment_scores))
    features_scaled = scaler.transform(features)

    # Split into training (80%) and test (20%)
    train_size = int(len(features_scaled) * 0.8)
    train, test = features_scaled[:train_size], features_scaled[train_size:]

    X_test, y_test = [], []
    for i in range(lookback, len(test)):
        X_test.append(test[i - lookback:i])
        y_test.append(test[i][0])  # Predicting Close Price

    X_test, y_test = np.array(X_test), np.array(y_test)

    # Predict prices
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform predictions and actual values
    y_pred = scaler.inverse_transform(np.c_[y_pred_scaled, np.zeros(len(y_pred_scaled))])[:, 0]
    y_test_actual = scaler.inverse_transform(np.c_[y_test, np.zeros(len(y_test))])[:, 0]

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    r2 = r2_score(y_test_actual, y_pred)

    print(f"📊 Evaluation of Saved LSTM Model for {stock_name}:")
    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📈 MAE: {mae:.2f}")
    print(f"📊 MAPE: {mape:.2f}%")
    print(f"📏 R² Score: {r2:.4f}")

    return mae, rmse, mape, r2

# Example Usage:
stock_name = "RELIANCE.NS"
evaluate_saved_lstm(stock_name)

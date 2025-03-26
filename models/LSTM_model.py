import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import os

# Load dataset
df = pd.read_csv("data/merged_data.csv", parse_dates=["Date"])

def prepare_lstm_data(stock_name, lookback=90):
    """
    Prepares training data for LSTM model by scaling and creating sequences.
    """
    stock_data = df[df["Stock"] == stock_name].sort_values("Date")

    if stock_data.empty:
        print(f"❌ No data found for {stock_name}")
        return None, None, None

    close_prices = stock_data["Close"].values.reshape(-1, 1)
    sentiment_scores = stock_data["Sentiment_Score"].values.reshape(-1, 1)  

    # Combine close prices and sentiment score
    features = np.hstack((close_prices, sentiment_scores))

    # Normalize data
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Save scaler
    joblib.dump(scaler, f"models/scaler_{stock_name}.pkl")

    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - lookback):
        X.append(features_scaled[i:i+lookback])
        y.append(features_scaled[i+lookback][0])  # Predicting Close Price

    return np.array(X), np.array(y), scaler

def train_lstm(stock_name, lookback=90, epochs=200, batch_size=64):
    """
    Trains an optimized LSTM model on the stock's historical prices and saves it.
    """
    X, y, scaler = prepare_lstm_data(stock_name, lookback)
    
    if X is None:
        return None

    # Define LSTM model
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True, input_shape=(lookback, 2))),
        Dropout(0.4),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")  # Linear activation for regression
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.0))

    early_stop = EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=7, min_lr=1e-6)

    # Train model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop, reduce_lr])

    # Save model
    model.save(f"models/lstm_{stock_name}.h5")
    print(f"✅ Optimized LSTM model trained and saved for {stock_name}")

    return model

def predict_next_day_price_lstm(stock_name, lookback=90):
    """
    Loads the trained LSTM model and predicts the next day's closing price.
    """
    try:
        model = tf.keras.models.load_model(f"models/lstm_{stock_name}.h5", compile=False)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.0))  # Fix metrics issue
        scaler = joblib.load(f"models/scaler_{stock_name}.pkl")
    except Exception:
        print(f"⚠️ No saved model found for {stock_name}. Training a new one...")
        model = train_lstm(stock_name, lookback)
        scaler = joblib.load(f"models/scaler_{stock_name}.pkl")

    if model is None:
        print(f"❌ Could not load or train LSTM model for {stock_name}")
        return None

    stock_data = df[df["Stock"] == stock_name].sort_values("Date")

    # Get last 'lookback' days of data
    close_prices = stock_data["Close"].values.reshape(-1, 1)
    sentiment_scores = stock_data["Sentiment_Score"].values.reshape(-1, 1)
    last_sequence = np.hstack((close_prices, sentiment_scores))[-lookback:]

    # Normalize data
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_scaled = last_sequence_scaled.reshape(1, lookback, 2)

    # Predict next day's price
    predicted_price_scaled = model.predict(last_sequence_scaled)[0][0]

    # Inverse transform to get the actual price
    predicted_price = scaler.inverse_transform([[predicted_price_scaled, 0]])[0][0]

    print(f"📈 Predicted closing price for {stock_name} (next day): ₹{predicted_price:.2f}")
    return predicted_price

# Example usage:
stock_name = "HDFCBANK.NS"
predict_next_day_price_lstm(stock_name)
 
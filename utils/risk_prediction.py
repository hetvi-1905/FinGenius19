import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    return data[['Close']]

def train_arima(data):
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    return forecast

def train_lstm(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return model, scaler

def predict_risk(data, lstm_model, scaler):
    last_60_days = data['Close'].values[-60:]  # ✅ Use 'Close' column explicitly
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))  # Reshape needed
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = lstm_model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))  # ✅ Fix reshape

    volatility = np.std(data['Close'].pct_change().dropna()).item()  # ✅ Ensure it's a float, not Series
    risk_score = "Low" if volatility < 0.02 else "Moderate" if volatility < 0.05 else "High"

    return predicted_price[0][0], risk_score

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ")
    data = fetch_stock_data(ticker)
    arima_forecast = train_arima(data)
    lstm_model, scaler = train_lstm(data)
    predicted_price, risk_score = predict_risk(data, lstm_model, scaler)
    
    print(f"Predicted Price (LSTM): {predicted_price}")
    print(f"Risk Score: {risk_score}")
    
    plt.figure(figsize=(10,5))
    plt.plot(data.index[-10:], arima_forecast, label='ARIMA Forecast', linestyle='dashed')
    plt.legend()
    plt.show()

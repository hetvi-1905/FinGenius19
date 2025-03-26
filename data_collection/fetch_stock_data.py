import yfinance as yf
import pandas as pd
import os

def get_nifty50_stocks():
    return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "WAAREEENER.NS"]  # Add all 50 stocks

def fetch_stock_data(stock_list, start="2020-01-01", end="2025-03-12"):
    if not os.path.exists("data"):
        os.makedirs("data")  # Ensure data folder exists

    for stock in stock_list:
        print(f"Fetching data for {stock}...")
        df = yf.download(stock, start=start, end=end, interval='1d')

        if df.empty:
            print(f"⚠️ No data found for {stock}. Skipping...")
            continue

        df.reset_index(inplace=True)  # Reset index to get 'Date' column
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Keep only necessary columns

        # **Fix: Rename columns properly**
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']  # Remove stock name in header

        # **Fix: Save with proper headers**
        file_path = f"data/{stock}.csv"
        df.to_csv(file_path, index=False)  # Save correctly

        print(f"✅ Data saved for {stock} -> {file_path}")

if __name__ == "__main__":
    stocks = get_nifty50_stocks()
    fetch_stock_data(stocks)

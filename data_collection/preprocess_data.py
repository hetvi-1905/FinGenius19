# # import pandas as pd
# # import os

# # def get_nifty50_stocks():
# #     return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]  # Add all 50 stocks

# # def load_stock_data(stock):
# #     file_path = f"data/{stock}.csv"
# #     if os.path.exists(file_path):
# #         df = pd.read_csv(file_path, usecols=["Date", "Close"], parse_dates=["Date"])
# #         df.rename(columns={"Close": "Stock_Price"}, inplace=True)
# #         df["Stock"] = stock
# #         return df
# #     return None

# # def load_sentiment_data():
# #     file_path = "data/news_sentiment.csv"
# #     if os.path.exists(file_path):
# #         return pd.read_csv(file_path)
# #     return None

# # def merge_data():
# #     stocks = get_nifty50_stocks()
# #     all_data = []
    
# #     sentiment_df = load_sentiment_data()
# #     if sentiment_df is None:
# #         print("Sentiment data not found!")
# #         return None
    
# #     for stock in stocks:
# #         stock_df = load_stock_data(stock)
# #         if stock_df is not None:
# #             sentiment_score = sentiment_df.loc[sentiment_df["Stock"] == stock, "Sentiment"].values
# #             stock_df["Sentiment_Score"] = sentiment_score[0] if len(sentiment_score) > 0 else 0
# #             all_data.append(stock_df)
    
# #     final_df = pd.concat(all_data, ignore_index=True)
# #     final_df.to_csv("data/merged_data.csv", index=False)
# #     print("Merged dataset saved successfully!")
# #     return final_df

# # if __name__ == "__main__":
# #     merge_data()

# import pandas as pd

# # Load stock price data
# stock_files = ["data/RELIANCE.NS.csv", "data/TCS.NS.csv", "data/INFY.NS.csv", "data/HDFCBANK.NS.csv"]
# stock_data = pd.concat([pd.read_csv(f) for f in stock_files], keys=stock_files)

# # Load sentiment data
# sentiment_data = pd.read_csv("data/news_sentiment.csv")

# # Ensure date format consistency
# stock_data["Date"] = pd.to_datetime(stock_data["Date"])
# sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"])

# # Merge based on Date & Stock
# merged_df = stock_data.merge(sentiment_data, on=["Date", "Stock"], how="left")

# # Fill missing sentiment with neutral (0)
# merged_df["Sentiment_Score"].fillna(0, inplace=True)

# # Save final merged data
# merged_df.to_csv("data/merged_data.csv", index=False)

# print("✅ Merging completed! Data saved to 'data/merged_data.csv'")
# -------------------------------------------------------
# import pandas as pd
# import os

# def get_nifty50_stocks():
#     return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]

# def preprocess_data():
#     stocks = get_nifty50_stocks()
#     merged_data = []

#     for stock in stocks:
#         stock_file = f"data/{stock}.csv"
#         sentiment_file = "data/news_sentiment.csv"

#         if not os.path.exists(stock_file) or not os.path.exists(sentiment_file):
#             print(f"Skipping {stock}: Missing data file.")
#             continue

#         # ✅ Load stock price data
#         stock_df = pd.read_csv(stock_file)
#         stock_df = stock_df[['Date', 'Close']].rename(columns={'Close': 'Stock_Price'})
#         stock_df['Stock'] = stock  # Add stock column for merging

#         # ✅ Load sentiment data
#         sentiment_df = pd.read_csv(sentiment_file)

#         # ✅ Merge stock price & sentiment data on Date & Stock
#         merged_df = pd.merge(stock_df, sentiment_df, on=['Date', 'Stock'], how='left')

#         # ✅ Fill missing sentiment scores with 0 (neutral)
#         merged_df['Sentiment_Score'].fillna(0, inplace=True)

#         merged_data.append(merged_df)

#     # ✅ Combine all stocks into one DataFrame
#     final_df = pd.concat(merged_data, ignore_index=True)

#     # ✅ Save merged data
#     final_df.to_csv("data/merged_data.csv", index=False)
#     print("✅ Saved merged_data.csv successfully!")

# if __name__ == "__main__":
#     preprocess_data()
# -------------------------------------------------------
import pandas as pd
import os

def get_nifty50_stocks():
    """Returns a list of selected NIFTY 50 stock symbols"""
    return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS","WAAREEENER.NS"]

def preprocess_data():
    stocks = get_nifty50_stocks()
    merged_data = []
    sentiment_file = "data/news_sentiment.csv"

    if not os.path.exists(sentiment_file):
        print("❌ Error: Sentiment file is missing!")
        return
    
    # ✅ Load sentiment data
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])  # Ensure Date is in datetime format

    for stock in stocks:
        stock_file = f"data/{stock}.csv"

        if not os.path.exists(stock_file):
            print(f"⚠️ Skipping {stock}: Missing stock file.")
            continue

        # ✅ Load stock price data
        stock_df = pd.read_csv(stock_file)

        # Ensure required columns exist
        required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(stock_df.columns):
            print(f"⚠️ Skipping {stock}: Missing required columns.")
            continue

        stock_df = stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={'Close': 'Close'})
        stock_df['Stock'] = stock  # Add stock column for merging
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])  # Convert Date to datetime
        
        # ✅ Merge stock price & sentiment data
        merged_df = pd.merge(stock_df, sentiment_df, on=['Date', 'Stock'], how='left')

        # ✅ Fill missing sentiment scores with 0 (neutral sentiment)
        merged_df['Sentiment_Score'].fillna(0, inplace=True)

        merged_data.append(merged_df)

    if not merged_data:
        print("❌ No data to merge! Check file availability.")
        return

    # ✅ Combine all stocks into one DataFrame
    final_df = pd.concat(merged_data, ignore_index=True)

    # ✅ Save merged data
    output_file = "data/merged_data.csv"
    final_df.to_csv(output_file, index=False)
    print(f"✅ Successfully saved merged data: {output_file}")

if __name__ == "__main__":
    preprocess_data()


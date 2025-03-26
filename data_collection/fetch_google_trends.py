# # # from pytrends.request import TrendReq
# # # import pandas as pd

# # # def get_nifty50_stocks():
# # #     return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]  # Add all 50 stocks


# # # def fetch_google_trends():
# # #     pytrends = TrendReq(hl='en-US', tz=360)
# # #     stocks = get_nifty50_stocks()
# # #     trends_data = {}
# # #     for stock in stocks:
# # #         pytrends.build_payload([stock], timeframe="today 5-y", geo="IN")
# # #         data = pytrends.interest_over_time()
# # #         trends_data[stock] = data[stock] if stock in data else 0
# # #     df = pd.DataFrame(trends_data)
# # #     df.to_csv("data/google_trends.csv")

# # # if __name__ == "__main__":
# # #     fetch_google_trends()

# # from pytrends.request import TrendReq
# # import pandas as pd
# # import time
# # import os

# def get_nifty50_stocks():
#     return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN"]  # Use stock names without .NS for Google Trends

# # def fetch_google_trends(stock_list, timeframe="today 5-y"):
# #     pytrends = TrendReq(hl='en-US', tz=330)
# #     if not os.path.exists("data"):
# #         os.makedirs("data")
    
# #     trends_data = {}
# #     for stock in stock_list:
# #         try:
# #             print(f"Fetching Google Trends data for {stock}...")
# #             pytrends.build_payload([stock], cat=0, timeframe=timeframe, geo='IN', gprop='')
# #             df = pytrends.interest_over_time()
# #             if not df.empty:
# #                 df.to_csv(f"data/{stock}_trends.csv")
# #                 trends_data[stock] = df
# #             time.sleep(20)  # Avoid hitting API rate limits
# #         except Exception as e:
# #             print(f"Error fetching trends for {stock}: {e}")
# #     return trends_data

# # if __name__ == "__main__":
# #     stocks = get_nifty50_stocks()
# #     fetch_google_trends(stocks)
# #     print("Google Trends data fetching completed.")
# import time
# from pytrends.request import TrendReq

# def fetch_google_trends(stock_list, timeframe="today 5-y"):
#     pytrends = TrendReq(hl='en-US', tz=330)
#     trends_data = {}

#     for stock in stock_list:
#         for attempt in range(3):  # Retry up to 3 times
#             try:
#                 print(f"Fetching Google Trends data for {stock} (Attempt {attempt+1})...")
#                 pytrends.build_payload([stock], cat=0, timeframe=timeframe, geo='IN', gprop='')
#                 df = pytrends.interest_over_time()

#                 if not df.empty:
#                     df.to_csv(f"data/{stock}_trends.csv")
#                     trends_data[stock] = df
#                 time.sleep(30)  # Increased delay
#                 break  # Exit retry loop if successful

#             except Exception as e:
#                 print(f"Error fetching trends for {stock}: {e}")
#                 time.sleep(60)  # Wait before retrying
#     return trends_data
# if __name__ == "__main__":
#     stocks = get_nifty50_stocks()
#     fetch_google_trends(stocks)
#     print("Google Trends data fetching completed.")
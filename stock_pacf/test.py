import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download stock data
ticker = 'AAPL'  # Example ticker for Apple Inc.
stock_data = yf.download(ticker, start='2020-01-01', end='2023-12-31')

# Display the first few rows of the data
print(stock_data.head())

# Plot the stock closing prices
plt.figure(figsize=(10, 5))
plt.plot(stock_data['Close'], label='Closing Price')
plt.title(f'{ticker} Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

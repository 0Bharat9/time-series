import yfinance as yf
import numpy as np


def cleaning():
    tickerSymbol = 'AAPL'
    stock_data = yf.download(tickerSymbol,
                             start='2021-01-01', end='2021-04-01')
    returns = stock_data.pct_change().dropna()
    return stock_data, returns


def stationarity(alias):
    first_diffs = alias.Close.values[1:] - alias.Close.values[:-1]
    first_diffs = np.concatenate([first_diffs, [0]])
    alias['FirstDifference'] = first_diffs
    return alias

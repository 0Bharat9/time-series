from statsmodels.tsa.stattools import adfuller
from data_cleaning import cleaning, stationarity
from plot import stock_plot, day_stock_plot


def adf_test_stock(series):
    result = adfuller(series.Close)
    print('ADF statistics:%f' % result[0])
    print('p-value: %f' % result[1])


def adf_test_FirstDiff(series):
    result = adfuller(series.FirstDifference)
    print('ADF statistics:%f' % result[0])
    print('p-value: %f' % result[1])


tickerDf = cleaning()
tickerDf2 = stationarity(tickerDf)

stock_plot(tickerDf)
adf_test_stock(tickerDf)
day_stock_plot(tickerDf2)
adf_test_FirstDiff(tickerDf2)

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_cleaning import cleaning, stationarity


tickerDf1 = cleaning()


def stock_plot(alias):
    plt.figure(figsize=(10, 4))
    plt.plot(alias.Close)
    plt.title('Stock Price over Time (AAPL)', fontsize=20)
    plt.ylabel('Price', fontsize=16)
    for year in range(2021, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'),
                    color='k', linestyle='--', alpha=0.2)
    plt.show()


def acf_plot(alias):
    plot_acf(alias)
    plt.show()


def pacf_plot(alias):
    plot_pacf(alias)
    plt.show()


def day_stock_plot(alias):
    plt.figure(figsize=(10, 4))
    plt.plot(alias.FirstDifference)
    plt.title('First Difference over Time (AAPL)', fontsize=20)
    plt.ylabel('Price Difference', fontsize=16)
    for year in range(2015, 2023):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'),
                    color='k', linestyle='--', alpha=0.2)
    plt.show()


def acf_plot2(alias):
    plot_acf(alias)
    plt.show()


def pacf_plot2(alias):
    plot_pacf(alias)
    plt.show()


if __name__ == "__main__":
    tickerDf2 = stationarity(tickerDf1)
    stock_plot(tickerDf1)
    day_stock_plot(tickerDf2)
    acf_plot(tickerDf2)
    pacf_plot(tickerDf2)

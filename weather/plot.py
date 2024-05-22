import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_cleaning import fetcher

df_weather = fetcher()


def basic_plot(alias, label):
    plt.figure(figsize=(10, 4))
    plt.plot(alias.index, alias[label])
    plt.title('Average temperature over time (New York)', fontsize=20)
    plt.ylabel('Temperature', fontsize=16)
    plt.show()


def acf_plot(alias):
    plot_acf(alias.tavg, lags=100)
    plt.show()


def pacf_plot(alias):
    plot_pacf(alias.tavg, lags=50)
    plt.show()


basic_plot(df_weather, 'tavg')
acf_plot(df_weather)
pacf_plot(df_weather)

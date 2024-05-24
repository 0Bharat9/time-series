import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_cleaning import fetcher, stationarity

df_weather = fetcher()


def basic_plot(alias, label):
    plt.figure(figsize=(10, 4))
    plt.plot(alias.index, alias[label])
    plt.title('Average temperature over time (New York)', fontsize=20)
    plt.ylabel('Temperature', fontsize=16)
    plt.show()


df_weather2 = stationarity(df_weather)


def day_temp_avg(alias):
    plt.figure(figsize=(10, 4))
    plt.plot(alias.FirstDifference)
    plt.title('Average temp diff for each day (New York)', fontsize=20)
    plt.ylabel('Temperature difference', fontsize=16)
    plt.show()


def acf_plot(alias):
    plot_acf(alias.FirstDifference, lags=100)
    plt.show()


def pacf_plot(alias):
    plot_pacf(alias.FirstDifference, lags=50)
    plt.show()


if __name__ == '__main__':
    basic_plot(df_weather, 'tavg')
    day_temp_avg(df_weather2)
    acf_plot(df_weather2)
    pacf_plot(df_weather2)

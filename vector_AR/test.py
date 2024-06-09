import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

data = pd.read_csv('./ice_cream_vs_heater.csv',
                   parse_dates=[0], index_col=0, date_format='%Y-%m')

data = data.asfreq(pd.infer_freq(data.index))
data = data.dropna()

heater_series = data.heater

print(heater_series)


def trend_plot(series):
    plt.figure(figsize=(10, 4))
    plt.plot(series, color='red')
    plt.ylabel('search frequency for heater', fontsize=16)

    for year in range(2004, 2021):
        plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)
    plt.show()


trend_plot(heater_series)

avg, dev = heater_series.mean(), heater_series.std()

heater_series = (heater_series-avg)/dev

trend_plot(heater_series)

heater_series = heater_series.diff().dropna()
trend_plot(heater_series)

annual_volatility = heater_series.groupby(heater_series.index.year).std()
print(annual_volatility)

heater_annual_vol = heater_series.index.map(
    lambda d: annual_volatility.loc[d.year])
print(heater_annual_vol)

heater_series = heater_series/heater_annual_vol

print(heater_series)

trend_plot(heater_series)

month_avgs = heater_series.groupby(heater_series.index.month).mean()
print(month_avgs)

heater_month_avg = heater_series.index.map(lambda d: month_avgs.loc[d.month])

print(heater_month_avg)

heater_series = heater_series-heater_month_avg

trend_plot(heater_series)


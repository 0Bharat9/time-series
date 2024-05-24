from statsmodels.tsa.stattools import adfuller
from data_cleaning import fetcher, stationarity
from plot import basic_plot, day_temp_avg


def adf_test_temp(series):
    result = adfuller(series.tavg)
    print('ADF statistics:%f' % result[0])
    print('p-value: %f' % result[1])


def adf_test_day_temp_avg(series):
    result = adfuller(series.FirstDifference)
    print('ADF statistics:%f' % result[0])
    print('p-value: %f' % result[1])


df_weather = fetcher()
df_weather2 = stationarity(df_weather)

basic_plot(df_weather, 'tavg')
adf_test_temp(df_weather)
day_temp_avg(df_weather2)
adf_test_day_temp_avg(df_weather2)

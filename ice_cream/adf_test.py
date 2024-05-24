from statsmodels.tsa.stattools import adfuller
from data_cleaning import cleaner
from plot import production_plot

df_ice_cream = cleaner()


def adf_test(series):
    results = adfuller(series.production)
    print('Statistics: %f' % results[0])
    print('p-value: %f' % results[1])


production_plot(df_ice_cream)

adf_test(df_ice_cream)

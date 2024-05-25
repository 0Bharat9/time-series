import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from data_cleaning import cleaner
from plot import plot_error, comparison_plot

warnings.filterwarnings("ignore")


df_ice_cream = cleaner()

train_part = datetime(2016, 12, 1)
test_part = datetime(2019, 12, 1)

train_data = df_ice_cream[:train_part]
test_data = df_ice_cream[train_part + timedelta(days=1):test_part]

# rolling forecast origin
# train on monthes 1,2,.....,k-2 -> predict month k-1
# add predicted month into data as it happens and predict the next month so on
# average all predictions
df_ice_cream = df_ice_cream.asfreq('MS')
test_data = test_data.asfreq('MS')
predictions_rolling = pd.Series(index=test_data.index)

for end_date in test_data.index:
    train_data = df_ice_cream[:end_date - pd.offsets.MonthEnd(1)]
    model = ARIMA(train_data, order=(3, 0, 0))
    model_fit = model.fit()
    pred = model_fit.predict(end_date)
    predictions_rolling.loc[end_date] = pred.loc[end_date]

error_rolling = test_data.squeeze() - predictions_rolling.squeeze()
plot_error(error_rolling, 2017, 2021)
comparison_plot(test_data, predictions_rolling, 2017, 2020)
print(np.sqrt(np.mean(error_rolling**2)))


#rolling forecast for moving average model
predictions_rolling = pd.Series(index=test_data.index)

for end_date in test_data.index:
    train_data = df_ice_cream[:end_date - pd.offsets.MonthEnd(1)]
    model = ARIMA(train_data, order=(0, 0, 3))
    model_fit = model.fit()
    pred = model_fit.predict(end_date)
    predictions_rolling.loc[end_date] = pred.loc[end_date]

error_rolling = test_data.squeeze() - predictions_rolling.squeeze()
plot_error(error_rolling, 2017, 2021)
comparison_plot(test_data, predictions_rolling, 2017, 2020)
print(np.sqrt(np.mean(error_rolling**2)))

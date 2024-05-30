import pandas as pd
import numpy as np
import warnings
from datetime import timedelta, datetime
from time import time
from statsmodels.tsa.arima.model import ARIMA
from data_cleaning import cleaner
from plot import production_plot, acf_plot, pacf_plot, plot_error, comparison_plot

warnings.filterwarnings("ignore")

df_ice_cream = cleaner()

production_plot(df_ice_cream, 'production', 2010, 2021)
acf_plot(df_ice_cream)
pacf_plot(df_ice_cream)

# based on pacf we can start with an ar(1),ar(2) or ar(3) models
train_part = datetime(2016, 12, 1)
test_part = datetime(2019, 12, 1)

train_data = df_ice_cream[:train_part]
test_data = df_ice_cream[train_part + timedelta(days=1):test_part]

print(train_data)
model = ARIMA(train_data, order=(15, 0, 0))

start = time()
model_fit = model.fit()
end = time()
print('Model fitting time:', end-start)

print(model_fit.summary())

pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
preds = model_fit.predict(start=pred_start_date, end=pred_end_date)
error = test_data.squeeze() - preds.squeeze()

plot_error(error, 2017, 2021)
comparison_plot(test_data, preds, 2017, 2020)
print(np.sqrt(np.mean(error**2)))

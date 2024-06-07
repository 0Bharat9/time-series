import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import time
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from data_cleaning import cleaner
from plot import production_plot, acf_plot, pacf_plot, comparison_plot, plot_error

df_ice_cream = cleaner()

df_ice_cream = df_ice_cream.asfreq(pd.infer_freq(df_ice_cream.index))

production_plot(df_ice_cream, 'production', 2010, 2021)

train_end = datetime(2016, 12, 1)
test_end = datetime(2019, 12, 1)

train_data = df_ice_cream[:train_end]
test_data = df_ice_cream[train_end+timedelta(days=1):test_end]

# plotting acf and pacf bar plots
lags = 20

acf_value = acf(df_ice_cream)
pacf_value = pacf(df_ice_cream)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.bar(range(lags), acf_value[:lags])
plt.title('ACF')
plt.subplot(2, 1, 2)
plt.bar(range(lags), pacf_value[:lags])
plt.title('PACF')
plt.show()


acf_plot(df_ice_cream)
pacf_plot(df_ice_cream)

model = SARIMAX(train_data, order=(0, 0, 0), seasonal_order=(1, 0, 1, 12))

start = time()
model_fit = model.fit()
end = time()
print("time taken to fit model:", end-start)
print(model_fit.summary())

start_date = test_data.index[0]
end_date = test_data.index[-1]

preds = model_fit.predict(start=start_date, end=end_date)

error = test_data.squeeze() - preds.squeeze()

plot_error(error, 2017, 2021)
comparison_plot(df_ice_cream, preds, 2010, 2021)

print(np.sqrt(np.mean(error**2)))

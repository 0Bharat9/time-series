import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from data_cleaning import cleaner
from plot import production_plot, plot_error, comparison_plot

df_ice_cream = cleaner()

df_ice_cream = df_ice_cream.asfreq(pd.infer_freq(df_ice_cream.index))

firstDifference = df_ice_cream.diff()[1:]

production_plot(firstDifference, 'production', 2010, 2021)
acf_value = acf(firstDifference)
plt.bar(range(14), acf_value[:14])
plt.show()
pacf_value = pacf(firstDifference)
plt.bar(range(14), pacf_value[:14])
plt.show()

train_end = datetime(2016, 12, 1)
test_end = datetime(2019, 12, 1)

train_data = firstDifference[:train_end]
test_data = firstDifference[train_end+timedelta(days=1):test_end]


model = ARIMA(train_data, order=(13, 0, 21))

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
comparison_plot(test_data, preds, 2017, 2021)

print(np.sqrt(np.mean(error**2)))

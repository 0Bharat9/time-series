import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from datetime import datetime, timedelta

errors = np.random.normal(0, 1, 400)

date_index = pd.date_range(start='9/1/2022', end='1/1/2023')

mu = 50
series = []
for t in range(1, len(date_index)+1):
    series.append(mu + 0.4*errors[t-1] + 0.3*errors[t-2] + errors[t])

series = pd.Series(series, date_index)
series = series.asfreq(pd.infer_freq(series.index))

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.axhline(mu, linestyle='--', color='grey', alpha=0.2)
plt.show()


def calc_corr(series, lag):
    return stats.pearsonr(series[:-lag], series[lag:])[0]


acf_value = acf(series)
lags = 20
plt.bar(range(lags), acf_value[:lags])
plt.show()

pacf_value = pacf(series)
lags = 20
plt.bar(range(lags), pacf_value[:lags])
plt.show()

train_end = datetime(2022, 12, 30)
test_end = datetime(2023, 1, 1)

train_data = series[:train_end]
test_data = series[train_end + timedelta(days=1):test_end]

model = ARIMA(train_data, order=(0, 0, 2))

model_fit = model.fit()

print(model_fit.summary())

pred_start = test_data.index[0]
pred_end = test_data.index[-1]
preds = model_fit.predict(start=pred_start, end=pred_end)

error = test_data.squeeze() - preds.squeeze()

plt.figure(figsize=(10, 4))
plt.plot(series[-15:])
plt.plot(preds)
plt.legend(('Data', 'Prediction'), fontsize=16)
plt.show()

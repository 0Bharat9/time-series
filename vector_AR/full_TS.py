import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


def perform_adf(series):
    result = adfuller(series)
    print('Adf statistics:', result[0])
    print('p-value', result[1])


data = pd.read_csv('./original_series.csv')
print(data)

plt.figure(figsize=(10, 4))
plt.plot(data)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours since published', fontsize=16)

plt.yticks(np.arange(0, 50000, 10000), fontsize=14)
plt.ylabel('Views', fontsize=16)
plt.show()

mu = data['0'].mean()
sigma = data['0'].std()
print(mu)
print(sigma)

normalized_data = (data-mu)/sigma
print(normalized_data)

plt.figure(figsize=(10, 4))
plt.plot(normalized_data)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours since published', fontsize=16)

plt.yticks(np.arange(-3, 2), fontsize=14)
plt.ylabel('Normalized Views', fontsize=16)
plt.axhline(0, color='k', linestyle='--')
plt.show()

exp_data = np.exp(normalized_data)
plt.figure(figsize=(10, 4))
plt.plot(exp_data)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours since published', fontsize=16)

plt.yticks(np.arange(0, 3), fontsize=14)
plt.ylabel('exp_data Views', fontsize=16)
plt.show()

perform_adf(exp_data)

first_diff_data = exp_data.diff().dropna()

plt.figure(figsize=(10, 4))
plt.plot(first_diff_data)

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours since published', fontsize=16)

plt.yticks(np.arange(0.2, 0.3, 0.1), fontsize=14)
plt.ylabel('first_diff Views', fontsize=16)
plt.show()

perform_adf(first_diff_data)

plot_pacf(first_diff_data)
plt.show()

plot_acf(first_diff_data)
plt.show()

model = ARIMA(first_diff_data, order=(4, 0, 1))

model_fit = model.fit()

lags = 3
forecast_result = model_fit.get_forecast(steps=lags)
preds = forecast_result.predicted_mean
confidence_intervals = forecast_result.conf_int()

print(preds)
print(confidence_intervals)

# Plotting the forecast
plt.figure(figsize=(10, 4))
plt.plot(first_diff_data, label='Original Data')

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours since published', fontsize=16)
plt.yticks(np.arange(-0.2, 0.3, 0.1), fontsize=14)
plt.ylabel('First Diff Views', fontsize=16)

plt.plot(np.arange(len(data), len(data) + lags),
         preds, color='r', label='Forecast')
plt.fill_between(np.arange(len(data), len(data) + lags),
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='g', alpha=0.1)

plt.legend()
plt.show()


def get_orig_result(preds, series, mu, sigma):
    first_pred = sigma * \
        np.log(preds.iloc[0] + np.exp((series.iloc[-1] - mu) / sigma)) + mu
    orig_preds = [first_pred]

    for i in range(1, len(preds)):
        next_pred = sigma * \
            np.log(preds.iloc[i] + np.exp((orig_preds[-1] - mu) / sigma)) + mu
        orig_preds.append(next_pred)

    return np.array(orig_preds).flatten()


orig_preds = get_orig_result(preds, data['0'], mu, sigma)
orig_lower_bound = get_orig_result(
    confidence_intervals.iloc[:, 0], data['0'], mu, sigma)
orig_upper_bound = get_orig_result(
    confidence_intervals.iloc[:, 1], data['0'], mu, sigma)

plt.figure(figsize=(10, 4))
plt.plot(data['0'], label='Original Data')
plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours since published', fontsize=16)
plt.yticks(np.arange(0, 50000, 10000), fontsize=14)
plt.ylabel('Views', fontsize=16)
plt.plot(np.arange(len(data), len(data) + lags),
         orig_preds, color='r', label='Forecast')
plt.fill_between(np.arange(len(data), len(data) + lags),
                 orig_lower_bound, orig_upper_bound, color='g', alpha=0.1)
plt.legend()
plt.show()


def get_orig_result(preds, series, mu, sigma):
    first_pred = sigma*np.log(preds[0]+np.exp((series.iloc[-1]-mu)/sigma))+mu
    orig_preds = [first_pred]

    for i in range(len(preds[1:])):
        next_pred = sigma * \
            np.log(preds[i+1] + np.exp((orig_preds[-1]-mu)/sigma))+mu
        orig_preds.append(next_pred)

        return np.array(orig_preds).flatten()


orig_preds = get_orig_result(preds, data, mu, sigma)
orig_lower_bound = get_orig_result(
    confidence_intervals.iloc[:, 0], data, mu, sigma)
orig_upper_bound = get_orig_result(
    confidence_intervals.iloc[:, 1], data, mu, sigma)

plt.figure(figsize=(10, 4))
plt.plot(first_diff_data, label='Original Data')

plt.xticks(np.arange(0, 78, 6), fontsize=14)
plt.xlabel('Hours since published', fontsize=16)
plt.yticks(np.arange(0, 50000, 10000), fontsize=14)
plt.ylabel('Views', fontsize=16)

plt.plot(np.arange(len(data), len(data) + lags),
         preds, color='r', label='Forecast')
plt.fill_between(np.arange(len(data), len(data) + lags),
                 orig_lower_bound,
                 orig_upper_bound, color='g', alpha=0.1)

plt.legend()
plt.show()

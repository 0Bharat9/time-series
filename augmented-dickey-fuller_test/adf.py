import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller


def ar_process_generator(lags, coeff, length):
    coeff = np.array(coeff)
    # starting values
    series = [np.random.normal() for _ in range(lags)]

    for _ in range(length):
        # get previous values of the series, reverse
        prev_vals = series[-lags:][::-1]

        # get new value of time series
        new_val = np.sum(np.array(prev_vals)*coeff) + np.random.normal()

        series.append(new_val)

    return np.array(series)


def adf_test(series):
    result = adfuller(series)
    print('ADF statistics:%f' % result[0])
    print('p-value: %f' % result[1])


# example:- stationary ar(1)
ar1_process = ar_process_generator(1, [.5], 200)
plt.figure(figsize=(10, 4))
plt.plot(ar1_process)
plt.title('stationary ar(1) process', fontsize=16)
plt.show()
adf_test(ar1_process)

# example non stationary unit root ar(1)
testar1_process = ar_process_generator(1, [1], 500)
plt.figure(figsize=(10, 4))
plt.plot(testar1_process)
plt.title('non stationary ar(1) process', fontsize=16)
plt.show()

adf_test(testar1_process)

# exmaple more complex ar(2)

ar2_process = ar_process_generator(2, [.5, .3], 200)
plt.figure(figsize=(10, 4))
plt.plot(ar2_process)
plt.title('stationarity ar(2) process', fontsize=16)
plt.show()

adf_test(ar2_process)

testar2_process = ar_process_generator(2, [.5, 1], 200)
plt.figure(figsize=(10, 4))
plt.plot(testar2_process)
plt.title('non stationarity ar(2) process', fontsize=16)
plt.show()

adf_test(testar2_process)


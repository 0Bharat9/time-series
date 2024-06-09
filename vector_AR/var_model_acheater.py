import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import pearsonr
from statsmodels.tsa.api import VAR

data = pd.read_csv('./ac_vs_heater.csv',
                   parse_dates=[0], index_col=0, date_format='%Y-%m')
data = data.dropna()
data = data.asfreq(pd.infer_freq(data.index))
print(data)
print(data.dtypes)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Print the data to inspect
print(data)
avgs, devs = data.mean(), data.std()

for col in data.columns:
    data[col] = (data[col]-avgs.loc[col]) / devs.loc[col]

data = data.diff().dropna()

annual_volatility = data.groupby(data.index.year).std()

data['ac_annual_vol'] = data.index.map(
    lambda d: annual_volatility.loc[d.year, 'ac'])
data['heater_annual_vol'] = data.index.map(
    lambda d: annual_volatility.loc[d.year, 'heater'])

data['ac'] = data['ac']/data['ac_annual_vol']
data['heater'] = data['heater']/data['heater_annual_vol']

months_avgs = data.groupby(data.index.month).mean()

data['ac_month_avg'] = data.index.map(
    lambda d: months_avgs.loc[d.month, 'ac'])
data['heater_month_avg'] = data.index.map(
    lambda d: months_avgs.loc[d.month, 'heater'])

data['ac'] = data['ac'] - data['ac_month_avg']
data['heater'] = data['heater'] - data['heater_month_avg']

data = data.dropna()


plt.figure(figsize=(12, 6))
air_condition, = plt.plot(data['ac'])
heater, = plt.plot(data['heater'], color='red')
for year in range(2004, 2025):
    plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)

plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.ylabel('First Difference', fontsize=18)

plt.legend(['Air air_condition', 'Heater'], fontsize=16)
plt.show()

plot_pacf(data['heater'])
plt.show()


for lag in range(1, 14):
    ac_series = data['ac'].iloc[lag:]
    lagged_heater_series = data['heater'].iloc[:-lag]
    print('Lag: %s' % lag)
    print(pearsonr(ac_series, lagged_heater_series))
    print('-----------------------------')

data = data[['heater', 'ac']]

model = VAR(data)

model_fit = model.fit(maxlags=13)

print(model_fit.summary())

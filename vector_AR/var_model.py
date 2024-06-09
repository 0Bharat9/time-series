import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import pearsonr
from statsmodels.tsa.api import VAR
from data_cleaning import cleaner


df_ice_cream_heater = cleaner()

plt.figure(figsize=(12, 6))
ice_cream, = plt.plot(df_ice_cream_heater['ice cream'])
heater, = plt.plot(df_ice_cream_heater['heater'], color='red')

for year in range(2004, 2021):
    plt.axvline(datetime(year, 1, 1), linestyle='--', color='k', alpha=0.5)

plt.axhline(0, linestyle='--', color='k', alpha=0.3)
plt.ylabel('First Difference', fontsize=18)

plt.legend(['Ice Cream', 'Heater'], fontsize=16)
plt.show()

plot_pacf(df_ice_cream_heater['heater'])
plt.show()

print(df_ice_cream_heater)

for lag in range(1, 14):
    heater_series = df_ice_cream_heater['heater'].iloc[lag:]
    lagged_ice_cream_series = df_ice_cream_heater['ice cream'].iloc[:-lag]
    print('Lag: %s' % lag)
    print(pearsonr(heater_series, lagged_ice_cream_series))
    print('-----------------------------')

#fitting var model

df_ice_cream_heater = df_ice_cream_heater[['ice cream', 'heater']]

model = VAR(df_ice_cream_heater)

model_fit = model.fit(maxlags=13)

print(model_fit.summary())

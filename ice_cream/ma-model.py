import numpy as np
from datetime import datetime, timedelta
from time import time
from statsmodels.tsa.arima.model import ARIMA
from data_cleaning import cleaner
from plot import acf_plot, pacf_plot, production_plot, plot_error, comparison_plot

df_ice_cream = cleaner()

mu = np.mean(df_ice_cream.production)


train_part = datetime(2016, 12, 1)
test_part = datetime(2019, 12, 1)

train_data = df_ice_cream[:train_part]
test_data = df_ice_cream[train_part + timedelta(days=1):test_part]
production_plot(df_ice_cream, 'production', 2010, 2021)
acf_plot(df_ice_cream)
pacf_plot(df_ice_cream)

model = ARIMA(train_data, order=(0, 0, 21))
start = time()
model_fit = model.fit()
end = time()
print('time taken to fit model:', end-start)

print(model_fit.summary())

pred_start = test_data.index[0]
pred_end = test_data.index[-1]

preds = model_fit.predict(start=pred_start, end=pred_end)

error = test_data.squeeze() - preds.squeeze()

plot_error(error, 2017, 2021)
comparison_plot(test_data, preds, 2017, 2021)

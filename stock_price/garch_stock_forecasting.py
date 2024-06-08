import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf

start = datetime(2012, 1, 1)
end = datetime(2024, 6, 7)

tickerSymbol = 'NVDA'
data = yf.Ticker(tickerSymbol)

prices = data.history(start=start, end=end).Close
returns = 100*prices.pct_change().dropna()

plt.figure(figsize=(10, 4))
plt.plot(returns)
plt.ylabel('percentage return', fontsize=16)
plt.title('DIS RETURNS', fontsize=20)
plt.show()

plot_pacf(returns**2)
plt.show()

model = arch_model(returns, p=3, q=3)

model_fit = model.fit()
print(model_fit.summary())

# as we can see garch (3,3) is not significant as beta values are not significant
# therefore we will be using garch(2,2)

model = arch_model(returns, p=2, q=2)
model_fit = model.fit()
print(model_fit.summary())

predictions = []
test_size = 365*4

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    predictions.append(pred)
predictions = pd.Series(predictions, index=returns.index[-test_size:])
print(predictions)


def extract_forecast_data(forecast):
    mean_forecast = forecast.mean['h.1'][0]
    # Accessing the variance forecast for horizon1 = pred.variance.values[-1, :][0]
    variance_forecast = np.sqrt(forecast.variance['h.1'][0])
    return pd.Series({'mean_forecast': mean_forecast, 'variance_forecast': variance_forecast})


# Apply the function to the predictions Series
extracted_data = predictions.apply(extract_forecast_data)

# Display the extracted data
print(extracted_data.head())
plt.figure(figsize=(10, 4))
true, = plt.plot(returns[-test_size:])
preds, = plt.plot(extracted_data['mean_forecast'])
plt.title('Prediction -> Rolling Forcast', fontsize=20)
plt.legend(['True Return', 'Predicted Volatility'], fontsize=16)
plt.show()

plt.figure(figsize=(10, 4))
true, = plt.plot(returns[-test_size:])
preds, = plt.plot(extracted_data['variance_forecast'])
plt.title('Volatility Prediction -> Rolling Forcast', fontsize=20)
plt.legend(['True Return', 'Predicted Volatility'], fontsize=16)
plt.show()

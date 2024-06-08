import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from arch import arch_model

start = datetime(2012, 1, 1)
end = datetime(2024, 6, 6)

tickerSymbol = 'NVDA'
data = yf.Ticker(tickerSymbol)

prices = data.history(start=start, end=end).Close
returns = 100*prices.pct_change().dropna()

train = returns
model = arch_model(train, p=2, q=2)
model_fit = model.fit(disp='off')
pred = model_fit.forecast(horizon=14)
future_days = [returns.index[-1] + timedelta(days=i) for i in range(1, 15)]
pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_days)

plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title('Volatility for next 7 days', fontsize=20)
plt.show()

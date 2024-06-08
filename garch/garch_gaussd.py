import numpy as np
import matplotlib.pyplot as plt
import random
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# at = εt*sqrt(w + α1*at-1^2 + α2*at-2^2 + β1*σt-1^2 + β2*σt-2^2)
# a0,a1 ~ N(0,1)
# σ0 = 1, σ1 = 1
# εt ~ N(0,1)

# creating data
n = 1000
w = 0.5
alpha_1 = 0.1
alpha_2 = 0.2

beta_1 = 0.3
beta_2 = 0.4

test_size = int(n*0.1)

series = [random.gauss(0, 1), random.gauss(0, 1)]
vols = [1, 1]

for _ in range(n):
    new_vol = np.sqrt(w + alpha_1*series[-1]**2
                      + alpha_2*series[-2]**2 + beta_1*vols[-1]**2
                      + beta_2*vols[-2]**2)
    new_val = random.gauss(0, 1)*new_vol
    vols.append(new_vol)
    series.append(new_val)

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title('simulated garch(2,2) data', fontsize=20)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(vols)
plt.title('data volatility', fontsize=20)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.plot(vols, color='red')
plt.title('data and vlatility', fontsize=20)
plt.show()

plot_acf(np.array(series)**2)
plt.show()

plot_pacf(np.array(series)**2)
plt.show()

train, test = series[:-test_size], series[-test_size:]
model = arch_model(train, p=2, q=2)
model_fit = model.fit()
print(model_fit.summary())

preds = model_fit.forecast(horizon=500)

plt.figure(figsize=(10, 4))
true, = plt.plot(vols[-test_size:])
prediction, = plt.plot(np.sqrt(preds.variance.values[-1, :]))
plt.title('valatility predictions', fontsize=20)
plt.legend(['True Volatility', 'Predicted volatility'], fontsize=16)
plt.show()

# rolling forecast
rolling_preds = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_preds.append(np.sqrt(pred.variance.values[-1, :][0]))

plt.figure(figsize=(10, 4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(rolling_preds)
plt.title('Volatility prediction -> rolling forecast', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
plt.show()

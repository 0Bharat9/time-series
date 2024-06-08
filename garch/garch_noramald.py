import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


n = 1000
w = 0.5
alpha_1 = 0.1
alpha_2 = 0.2
beta_1 = 0.3
beta_2 = 0.4

series = [np.random.normal(0, 1), np.random.normal(0, 1)]
vols = [1, 1]

for i in range(n):
    new_vol = np.sqrt(w + alpha_1*series[-1]**2 + alpha_2*series[-2]**2
                      + beta_1*vols[-1]**2 + beta_2*vols[-2]**2)
    new_val = np.random.normal()*new_vol
    vols.append(new_vol)
    series.append(new_val)

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title('garch(2,2) data', fontsize=20)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(vols)
plt.title('volatility in data', fontsize=20)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.plot(vols, color='red')
plt.title('volatility and data', fontsize=20)
plt.show()

plot_acf(np.array(series)**2)
plt.show()
plot_pacf(np.array(series)**2)
plt.show()

test_size = int(n*0.1)

train_set, test_set = series[:-test_size], series[-test_size:]

model = arch_model(train_set, p=2, q=2)
model_fit = model.fit()
print(model_fit.summary())
preds = model_fit.forecast(horizon=500)

plt.figure(figsize=(10, 4))
true, = plt.plot(vols[-test_size:])
predictions, = plt.plot(np.sqrt(preds.variance.values[-1, :]))
plt.title("predicted values", fontsize=20)
plt.legend(['True volatility', 'Predicted volatility'], fontsize=16)
plt.show()


rolling_preds = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    preds = model_fit.forecast(horizon=1)
    rolling_preds.append(np.sqrt(preds.variance.values[-1, :][0]))

plt.figure(figsize=(10, 4))
true, = plt.plot(vols[-test_size:])
predictions, = plt.plot(rolling_preds)
plt.title("predicted values", fontsize=20)
plt.legend(['True volatility', 'Predicted volatility'], fontsize=16)
plt.show()

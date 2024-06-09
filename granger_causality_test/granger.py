import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# building a simple AR(1) time series

a1 = [0.1*np.random.normal()]
for _ in range(100):
    a1.append(0.5*a1[-1] + 0.1*np.random.normal())

# building a time series which is granger caused by a1

a2 = [var + 0.1*np.random.normal() for var in a1]

# adjust a1 and a2 lags
a1 = a1[3:]
a2 = a2[:-3]

plt.figure(figsize=(10, 4))
plt.plot(a1, color='r')
plt.plot(a2, color='b')

plt.legend(['a1', 'a2'], fontsize=16)
plt.show()

as_df = pd.DataFrame(columns=['a2', 'a1'], data=zip(a2, a1))
print(as_df)

gc_results = grangercausalitytests(as_df, 3)
print(gc_results)

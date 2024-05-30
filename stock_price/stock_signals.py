import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from plot import acf_plot, pacf_plot

warnings.filterwarnings('ignore')


def run_simulation(returns, prices, amount, order, thresh, verbose=False, plot=True):
    if type(order) == float:
        thresh = None
    current_holdings = False
    event_list = []
    initial_amount = amount

    # Ensure both returns and prices indices are timezone-naive
    returns.index = returns.index.tz_localize(None)
    prices.index = prices.index.tz_localize(None)

    # Going through dates
    for date, r in tqdm(returns.iloc[14:].items(), total=len(returns.iloc[14:])):
        # date is already the index, no need to convert again
        date = pd.to_datetime(date).tz_localize(
            None)  # Ensure the date is timezone-naive

        # If you are holding the stock, sell it
        if current_holdings:
            sell_price = prices.loc[date]
            current_holdings = False
            ret = (sell_price - buy_price) / buy_price
            amount = amount * (1 + ret)
            event_list.append(('s', date, ret))

            if verbose:
                print('Sold at $%s' % sell_price)
                print('Predicted Return: %s' % round(pred, 4))
                print('Actual Return: %s' % round(ret, 4))
            continue

        # Get data till before date
        current_data = returns[:date]
        if type(order) is tuple:
            # Fit model
            try:
                model = ARIMA(current_data, order=order)
                model_fit = model.fit()
                # Get prediction
                pred = model_fit.forecast(steps=1)[0]
                print(pred)
            except:
                pred = thresh - 1

        # If model predicts a high return then buy
        if (not current_holdings) and \
                ((type(order) == float and np.random.random() < order)
                 or (type(order) == tuple and pred > thresh)
                 or (order == 'last' and current_data[-1] > 0)):

            current_holdings = True
            buy_price = prices.loc[date]
            event_list.append(('b', date))
            if verbose:
                print('Bought at $%s' % buy_price)

    if verbose:
        print('Total amount: $%s' % round(amount, 2))

    # Graph
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(prices[14:])

        y_lims = (int(prices.min() * .95), int(prices.max() * 1.05))
        shaded_y_lims = int(prices.min() * .5), int(prices.max() * 1.5)

        for idx, event in enumerate(event_list):
            plt.axvline(event[1], color='k', linestyle='--', alpha=0.4)
            if event[0] == 's':
                color = 'green' if event[2] > 0 else 'red'
                plt.fill_betweenx(
                    range(*shaded_y_lims), event[1], event_list[idx - 1][1], color=color, alpha=0.1)

        total_return = round(100 * (amount / initial_amount - 1), 2)
        total_return = str(total_return)
        plt.title("%s Price Data\nThresh=%s\nTotal Amount: $%s\nTotal Return: %s" % (
            'AAPL', thresh, amount, total_return))
        plt.ylim(*y_lims)
        plt.show()

    return amount


tickerSymbol = 'MCFT'
data = yf.Ticker(tickerSymbol)

prices = data.history(start='2020-07-04', end='2020-09-10').Close
returns = prices.pct_change().dropna()
acf_plot(returns)
pacf_plot(returns)
# baseline model random buying
run_simulation(returns, prices, 100, 0.5, None, verbose=False)

# simulating random buying for 1500 times
final_amounts = [run_simulation(
    returns, prices, 100, 0.5, None, verbose=False, plot=False) for _ in range(1500)]

plt.figure(figsize=(15, 8))
sns.distplot(final_amounts)
plt.axvline(np.mean(final_amounts), color='k', linestyle='--')
plt.axvline(100, color='g', linestyle='--')
plt.title('Avg: $%s\nSD: $%s' % (round(np.mean(final_amounts), 2),
                                 round(np.std(final_amounts), 2)), fontsize=20)
plt.show()

# if last return is positive then buy
run_simulation(returns, prices, 100, 'last', None, verbose=False)

# AR(1) model
for thresh in [0, 0.001, 0.005]:
    run_simulation(returns, prices, 200, (1, 0, 0), thresh, verbose=True)

# AR(5) model
for thresh in [0, 0.001, 0.005]:
    run_simulation(returns, prices, 200, (1, 0, 0), thresh, verbose=True)

# ARMA(5,5) model
for thresh in [0, 0.001, 0.005]:
    run_simulation(returns, prices, 200, (1, 0, 0), thresh, verbose=True)

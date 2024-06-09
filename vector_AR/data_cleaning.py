import pandas as pd


def cleaner():
    data = pd.read_csv('./ice_cream_vs_heater.csv',
                       parse_dates=[0], index_col=0, date_format='%Y-%m')
    data = data.asfreq(pd.infer_freq(data.index))

    print(data.columns)
    avgs, devs = data.mean(), data.std()

    for col in data.columns:
        data[col] = (data[col]-avgs.loc[col]) / devs.loc[col]

    data = data.diff().dropna()

    annual_volatility = data.groupby(data.index.year).std()

    data['ice_cream_annual_vol'] = data.index.map(
        lambda d: annual_volatility.loc[d.year, 'ice cream'])
    data['heater_annual_vol'] = data.index.map(
        lambda d: annual_volatility.loc[d.year, 'heater'])

    data['ice cream'] = data['ice cream'] / data['ice_cream_annual_vol']
    data['heater'] = data['heater'] / data['heater_annual_vol']

    month_avgs = data.groupby(data.index.month).mean()

    data['ice_cream_month_avg'] = data.index.map(
        lambda d: month_avgs.loc[d.month, 'ice cream'])
    data['heater_month_avg'] = data.index.map(
        lambda d: month_avgs.loc[d.month, 'heater'])

    data['ice cream'] = data['ice cream'] - data['ice_cream_month_avg']
    data['heater'] = data['heater'] - data['heater_month_avg']

    return data

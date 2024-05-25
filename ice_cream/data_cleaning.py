import pandas as pd
from datetime import datetime


def cleaner():
    data = pd.read_csv('./ice_cream.csv')
    data.rename(columns={'DATE': 'date',
                         'IPN31152N': 'production'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data = data[pd.to_datetime('2010-01-01'):]
    return data

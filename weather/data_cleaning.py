import pandas as pd
import numpy as np
from meteostat import Point, Daily


def fetcher():
    location = Point(30.912, 75.8538)
    start = pd.Timestamp('2024-01-01')
    end = pd.Timestamp('2024-05-09')
    data = Daily(location, start, end)
    data = data.fetch()
    return data


def stationarity(alias):
    first_diffs = alias.tavg.values[1:] - alias.tavg.values[:-1]
    first_diffs = np.concatenate([first_diffs, [0]])
    alias['FirstDifference'] = first_diffs
    return alias

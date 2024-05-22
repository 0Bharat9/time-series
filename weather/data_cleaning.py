import pandas as pd
from meteostat import Point, Daily


def fetcher():
    location = Point(30.912, 75.8538)
    start = pd.Timestamp('2024-01-01')
    end = pd.Timestamp('2024-05-09')
    data = Daily(location, start, end)
    data = data.fetch()
    return data

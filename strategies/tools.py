import numpy as np


def find_first_(lst, value):
    """Return the index of the first element equal to value, or 'failed'."""
    ids = [i for i, x in enumerate(lst) if x == value]
    return ids[0] if ids else 'failed'


def find_all_(lst, value):
    """Return a list of all indices equal to value, or 'failed'."""
    ids = [i for i, x in enumerate(lst) if x == value]
    return ids if ids else 'failed'


def backtest_prepare(df, start_date):
    """Final preparation before backtesting: suppress entries on the last bar."""
    df.loc[df.index[-1], df.filter(like='entry_signal').columns] = False
    return df

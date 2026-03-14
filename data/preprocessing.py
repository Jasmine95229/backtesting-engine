import pandas as pd
import numpy as np
from functools import reduce


def abnormal_check(df: pd.DataFrame, filter_level: float):
    """Flag timestamps where any OHLC column has a price change exceeding filter_level.

    Checks both forward and backward percentage changes to catch isolated
    spikes that revert on the next bar.

    Args:
        df: DataFrame with price columns containing 'Open', 'High', 'Low', or 'Close'.
        filter_level: Threshold as a decimal (e.g. 0.2 = 20%).

    Returns:
        Original DataFrame with anomaly columns appended (for inspection).
    """
    cols = [col for col in df.columns if any(x in col for x in ['Open', 'High', 'Low', 'Close'])]
    for col in cols:
        prev_change = df[col].pct_change()
        next_change = df[col].pct_change(-1)
        df[f'{col}_anomaly'] = (prev_change.abs() > filter_level) | (next_change.abs() > filter_level)

    anomaly_cols = [f'{col}_anomaly' for col in cols]
    df['any_anomaly'] = df[anomaly_cols].any(axis=1)

    anomalous_times = df.index[df['any_anomaly']]
    if not anomalous_times.empty:
        print(f"Timestamps with >{filter_level * 100}% price anomaly:")
        print(anomalous_times)

    return df


def align_hist_price(tradable_time, hist_price):
    """Align historical prices to a generated tradable-time grid.

    Missing bars are forward-filled and marked with tradable=2
    so the engine can skip signal logic on stale data.

    Args:
        tradable_time: DataFrame from generate_time_list (has 'tradable' column).
        hist_price: Raw historical price DataFrame indexed by datetime.

    Returns:
        Combined DataFrame with tradable-time columns + aligned price columns.
    """
    valid_times = set(hist_price.index)

    hist_price = hist_price.sort_index()
    hist_price = hist_price[~hist_price.index.duplicated(keep="last")]

    hist_price_aligned = hist_price.reindex(tradable_time.index, method='ffill')
    was_filled = ~hist_price_aligned.index.isin(valid_times)

    df_combined = pd.concat([tradable_time.copy(), hist_price_aligned], axis=1)
    df_combined.loc[was_filled, 'tradable'] = 2

    return df_combined


def directions_hist_prices(hist_prices: list, directions: list):
    """Prefix all columns with direction names and merge into one DataFrame.

    For a single-asset strategy with both Long and Short directions,
    this creates Long_BidClose, Short_BidClose, Long_tradable, Short_tradable, etc.

    For multi-asset strategies (e.g. MarketNeutral), each direction can use
    a different asset's price data.

    Args:
        hist_prices: List of DataFrames (one per direction).
        directions: List of direction names (e.g. ['Long', 'Short']).

    Returns:
        Merged DataFrame with all columns prefixed by direction.
    """
    dfs = []
    for direction, hist_price in zip(directions, hist_prices):
        df = hist_price.copy()
        df.columns = [f'{direction}_{col}' for col in df.columns]
        dfs.append(df)

    all_hist_prices = reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'),
        dfs)

    return all_hist_prices


def recheck_open_tradable(df, directions: list):
    """Mark market-open bars as untradable if the open price gaps vs previous close.

    A gap at market open means the displayed open price may not reflect
    executable prices, so these bars are marked tradable=3.

    Args:
        df: DataFrame with direction-prefixed columns (Long_AskOpen, etc.).
        directions: List of direction names (e.g. ['Long', 'Short']).

    Returns:
        DataFrame with updated tradable flags.
    """
    for d in directions:
        condition_long = (
            (df[f'{d}_AskOpen'] <= df[f'{d}_AskClose'].shift(1)) &
            (df[f'{d}_BidOpen'] <= df[f'{d}_BidClose'].shift(1))
        )
        condition_short = (
            (df[f'{d}_AskOpen'] >= df[f'{d}_AskClose'].shift(1)) &
            (df[f'{d}_BidOpen'] >= df[f'{d}_BidClose'].shift(1))
        )

        if 'long' in d.lower():
            if 'Long_market_open' in df.columns:
                df['Long_tradable'] = np.where(
                    (df['Long_tradable'] == 1) & df['Long_market_open'] & ~condition_long,
                    3, df['Long_tradable'])
        else:
            if 'Short_market_open' in df.columns:
                df['Short_tradable'] = np.where(
                    (df['Short_tradable'] == 1) & df['Short_market_open'] & ~condition_short,
                    3, df['Short_tradable'])

    return df


def stats_hist_prices(hist_prices, start_date, end_date, max_stats):
    """Trim historical prices to the backtest date range, keeping lookback rows for indicators.

    When max_stats > 0, prepends that many rows before start_date so strategies
    can compute rolling statistics. Those prepended rows are marked tradable=4
    to prevent the engine from generating signals on them.

    Args:
        hist_prices: Full historical DataFrame with direction-prefixed tradable columns.
        start_date: Backtest start date string.
        end_date: Backtest end date string.
        max_stats: Number of lookback rows to keep before start_date (0 = no lookback).

    Returns:
        Trimmed DataFrame.
    """
    if max_stats == 0:
        hist_prices = hist_prices[hist_prices.index >= pd.to_datetime(start_date)]
        hist_prices = hist_prices[hist_prices.index <= pd.to_datetime(end_date)]
        return hist_prices

    if start_date:
        drop_backtest_data = hist_prices[hist_prices.index < pd.to_datetime(start_date)]
        keep_drop_backtest_data = drop_backtest_data.iloc[-max_stats:]
        hist_prices = hist_prices[hist_prices.index >= pd.to_datetime(start_date)]
        hist_prices = pd.concat([hist_prices, keep_drop_backtest_data]).sort_index()

    if end_date:
        hist_prices = hist_prices[hist_prices.index <= pd.to_datetime(end_date)]

    tradable_cols = [col for col in hist_prices.columns if col.endswith('_tradable')]
    hist_prices.iloc[:max_stats, hist_prices.columns.get_indexer(tradable_cols)] = 4

    return hist_prices


def remove_unused(df: pd.DataFrame, unused: list):
    """Drop columns whose names contain any of the given substrings."""
    return df.loc[:, ~df.columns.str.contains('|'.join(unused), case=False)]

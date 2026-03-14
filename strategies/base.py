"""Base class for strategies — provides the shared data loading pipeline.

Every strategy's __init__ repeats the same 6-step data pipeline:
    1. Load CSV prices via histPrices()
    2. Check for anomalous price spikes
    3. Generate a tradable-time grid
    4. Align prices to the grid (forward-fill gaps)
    5. Create direction-prefixed columns (Long_*, Short_*)
    6. Trim to the backtest date range

This base class extracts that shared pipeline so strategy subclasses
only need to implement:
    - prepare_backtest_data()  — compute indicators & signals
    - update_signal()          — per-bar signal logic during backtest
    - temp_matrices             — nested class for strategy-specific arrays
    - strat_record              — nested class (usually just inherits BacktestRecord_strat)
"""

from data.loader import histPrices
from data.preprocessing import (
    abnormal_check, align_hist_price, directions_hist_prices,
    recheck_open_tradable, stats_hist_prices
)
from data.time_utils import generate_time_list


class StrategyBase:
    """Shared data pipeline for all strategies.

    Subclasses should call super().__init__() to run the pipeline,
    then implement prepare_backtest_data() and update_signal().
    """

    def __init__(self, dates, Tickers, timeframe='H1',
                 position_directions=None, max_stats=0,
                 data_source=None, **kwargs):
        if position_directions is None:
            position_directions = ['Long', 'Short']

        self.dates = dates
        self.timeframe = timeframe
        self.Ticker = Tickers
        self.position_directions = position_directions
        self.data_source = data_source

        # --- Shared data pipeline ---
        histPrice = abnormal_check(
            histPrices(self.Ticker, self.timeframe, source=self.data_source), 0.2)

        generated_time_ls = generate_time_list(
            start_time=histPrice.index[0],
            end_time=histPrice.index[-1],
            timeframe=self.timeframe,
            ticker=self.Ticker)

        hist_prices = align_hist_price(generated_time_ls, histPrice)
        hist_prices = directions_hist_prices(
            [hist_prices] * len(position_directions), position_directions)
        hist_prices = recheck_open_tradable(hist_prices, position_directions)
        hist_prices = stats_hist_prices(hist_prices, dates[0], dates[1], max_stats)

        self.histPrices = hist_prices

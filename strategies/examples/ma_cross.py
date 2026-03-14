"""Moving Average Crossover strategy with adaptive exits.

Entry: MA crossover with slope confirmation, efficiency filter, and instability filter.
Exit: ATR-based trailing stop, ATR take-profit, max holding time, or signal reversal.

Filters:
    - Slope confirmation: short MA must slope in entry direction for N bars
    - Efficiency: slope difference must exceed threshold to avoid noise
    - Instability: suppress entry if opposing crossover occurred within lookback window
    - Extreme range: suppress entry after abnormally large candles
    - Session hours: optional time-of-day filter
    - ATR regime: optional minimum ATR filter
"""

import numpy as np
import pandas as pd

from engine.records import BacktestRecord_temp, BacktestRecord_strat
from strategies.tools import find_first_, find_all_, backtest_prepare


class MACross:

    def __init__(self, dates: tuple, Tickers: str,
                 ma_type: str = "ema", short_ma: int = 9, long_ma: int = 26,
                 TP: float = None, stop_PnL: bool = False,
                 leverage: float = 1.0, timeframe: str = 'M5',
                 direction: str = 'Both',
                 position_directions: list = None,
                 position_weights: list = None,
                 max_DirectionsPosition: list = None,
                 # Entry quality
                 extreme_range_window: int = 200, extreme_range_pct: float = 95.0,
                 extreme_range_lookback: int = 3,
                 eff_threshold: float = 0.0004, unstable_window: int = 50,
                 slope_confirm_bars: int = 2,
                 # Adaptive exits
                 atr_period: int = 14, initial_sl_atr: float = 3.0,
                 trail_activation_atr: float = 1.5, trail_distance_atr: float = 2.0,
                 tp_atr: float = None, max_holding_bars: int = None,
                 # Session & volatility filters
                 trade_start_hour: int = None, trade_end_hour: int = None,
                 min_atr_pct: float = None,
                 **kwargs):
        if position_directions is None:
            position_directions = ['Long', 'Short']
        if position_weights is None:
            position_weights = [1, 1]
        if max_DirectionsPosition is None:
            max_DirectionsPosition = [1, 1]

        self.dates = dates
        self.timeframe = timeframe
        self.Ticker = Tickers
        self.ma_type = ma_type.lower()
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.TP = TP
        self.leverage = leverage
        self.direction = direction
        self.position_directions = position_directions
        self.position_weights = position_weights
        self.max_DirectionsPosition = [1, 1]
        self.extreme_range_window = extreme_range_window
        self.extreme_range_pct = extreme_range_pct
        self.extreme_range_lookback = extreme_range_lookback
        self.last_confirmed_trend = 0
        self.next_confirmed_trend = 0
        self.eff_threshold = eff_threshold
        self.unstable_window = unstable_window
        self.slope_confirm_bars = slope_confirm_bars
        self.atr_period = atr_period
        self.initial_sl_atr = initial_sl_atr
        self.trail_activation_atr = trail_activation_atr
        self.trail_distance_atr = trail_distance_atr
        self.tp_atr = tp_atr
        self.max_holding_bars = max_holding_bars
        self.trade_start_hour = trade_start_hour
        self.trade_end_hour = trade_end_hour
        self.min_atr_pct = min_atr_pct
        self.data_source = kwargs.pop('data_source', None)

        # --- Data pipeline ---
        from data.loader import histPrices
        from data.preprocessing import (
            abnormal_check, align_hist_price, directions_hist_prices,
            recheck_open_tradable, stats_hist_prices
        )
        from data.time_utils import generate_time_list

        histPrice = abnormal_check(
            histPrices(self.Ticker, self.timeframe, source=self.data_source), 0.2)
        generated_time_ls = generate_time_list(
            start_time=histPrice.index[0], end_time=histPrice.index[-1],
            timeframe=self.timeframe, ticker=self.Ticker)
        hist_prices = align_hist_price(generated_time_ls, histPrice)
        hist_prices = directions_hist_prices([hist_prices, hist_prices], self.position_directions)
        hist_prices = recheck_open_tradable(hist_prices, self.position_directions)
        hist_prices = stats_hist_prices(hist_prices, self.dates[0], self.dates[1], 0)
        self.histPrices = hist_prices

    def prepare_backtest_data(self):
        data = self.histPrices.copy()

        # Default price mappings
        data['Long_entry_signal'] = False
        data['Long_entry_price'] = data['Long_AskHigh']
        data['Long_current_price'] = data['Long_BidClose']
        data['Long_exit_signal'] = False
        data['Long_exit_price'] = data['Long_BidLow']

        data['Short_entry_signal'] = False
        data['Short_entry_price'] = data['Short_BidLow']
        data['Short_current_price'] = data['Short_AskClose']
        data['Short_exit_signal'] = False
        data['Short_exit_price'] = data['Short_AskHigh']

        data['Leverage'] = self.leverage

        # --- Moving averages ---
        mid = (data['Long_AskClose'] + data['Long_BidClose']) / 2
        if self.ma_type == "ema":
            data['MA_short'] = mid.ewm(span=self.short_ma, adjust=False).mean()
            data['MA_long'] = mid.ewm(span=self.long_ma, adjust=False).mean()
        else:
            data['MA_short'] = mid.rolling(window=self.short_ma).mean()
            data['MA_long'] = mid.rolling(window=self.long_ma).mean()

        # Slope & efficiency
        data['short_slope'] = data['MA_short'].diff()
        data['long_slope'] = data['MA_long'].diff()
        short_slope_norm = data['short_slope'] / mid
        long_slope_norm = data['long_slope'] / mid
        slope_diff = short_slope_norm - long_slope_norm
        eff = slope_diff.abs() > self.eff_threshold

        # Crossover detection
        data['Crossup'] = (data['MA_short'] > data['MA_long']) & (data['MA_short'].shift(1) < data['MA_long'].shift(1)) & eff
        data['Crossdown'] = (data['MA_short'] < data['MA_long']) & (data['MA_short'].shift(1) > data['MA_long'].shift(1)) & eff

        # Slope confirmation
        if self.slope_confirm_bars > 0:
            slope_positive = (data['short_slope'] > 0).rolling(
                window=self.slope_confirm_bars, min_periods=self.slope_confirm_bars).min().astype(bool)
            slope_negative = (data['short_slope'] < 0).rolling(
                window=self.slope_confirm_bars, min_periods=self.slope_confirm_bars).min().astype(bool)
            data['Crossup'] = data['Crossup'] & slope_positive
            data['Crossdown'] = data['Crossdown'] & slope_negative

        # Delay by 1 bar (trade on next open)
        crossup_delay = data['Crossup'].shift(1).fillna(False)
        crossdown_delay = data['Crossdown'].shift(1).fillna(False)

        # Instability filter
        up_unstable = crossup_delay.shift(1).rolling(window=self.unstable_window, min_periods=1).sum()
        down_unstable = crossdown_delay.shift(1).rolling(window=self.unstable_window, min_periods=1).sum()

        data['Long_entry_signal'] = crossup_delay & (down_unstable == 0)
        data['Long_exit_signal'] = crossdown_delay
        data['Short_entry_signal'] = crossdown_delay & (up_unstable == 0)
        data['Short_exit_signal'] = crossup_delay

        # Extreme range filter
        candle_range = data['Long_AskHigh'] - data['Long_BidLow']
        range_threshold = candle_range.rolling(window=self.extreme_range_window).quantile(self.extreme_range_pct / 100.0)
        any_recent_extreme = (candle_range > range_threshold).rolling(
            window=self.extreme_range_lookback, min_periods=1).max().astype(bool)
        data['Long_entry_signal'] = data['Long_entry_signal'] & ~any_recent_extreme
        data['Short_entry_signal'] = data['Short_entry_signal'] & ~any_recent_extreme

        # ATR
        tr = pd.concat([
            data['Long_AskHigh'] - data['Long_BidLow'],
            (data['Long_AskHigh'] - mid.shift(1)).abs(),
            (data['Long_BidLow'] - mid.shift(1)).abs()
        ], axis=1).max(axis=1)
        data['ATR'] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # Session filter
        if self.trade_start_hour is not None:
            valid_hours = (data['Long_Hour'] >= self.trade_start_hour) & (data['Long_Hour'] < self.trade_end_hour)
            data['Long_entry_signal'] = data['Long_entry_signal'] & valid_hours
            data['Short_entry_signal'] = data['Short_entry_signal'] & valid_hours

        # ATR regime filter
        if self.min_atr_pct is not None:
            atr_sufficient = (data['ATR'] / mid) > self.min_atr_pct
            data['Long_entry_signal'] = data['Long_entry_signal'] & atr_sufficient
            data['Short_entry_signal'] = data['Short_entry_signal'] & atr_sufficient

        self.backtest_data = backtest_prepare(data, self.dates[0])

        self.signal_data = self.backtest_data[[
            'Long_entry_signal', 'Long_exit_signal', 'Long_entry_price',
            'Long_current_price', 'Long_exit_price',
            'Short_entry_signal', 'Short_exit_signal', 'Short_entry_price',
            'Short_current_price', 'Short_exit_price',
            'Leverage', 'Crossup', 'Crossdown',
            'ATR', 'Long_BidHigh', 'Short_AskLow'
        ]]

        return self.backtest_data

    class temp_matrices(BacktestRecord_temp):
        def __init__(self, position_directions=None, position_weights=None, max_DirectionsPosition=None):
            if position_directions is None:
                position_directions = ['Long', 'Short']
            if position_weights is None:
                position_weights = [1, 1]
            if max_DirectionsPosition is None:
                max_DirectionsPosition = [1, 1]
            super().__init__(position_directions, position_weights, max_DirectionsPosition)
            rows = max(max_DirectionsPosition)
            cols = len(max_DirectionsPosition)
            self.past_trade = np.zeros((rows, cols))
            self.sl_price = np.zeros((rows, cols))
            self.highest_favorable_price = np.zeros((rows, cols))
            self.trail_active = np.full((rows, cols), False, dtype=bool)
            self.tp_exit_price = np.zeros((rows, cols))

    class strat_record(BacktestRecord_strat):
        def __init__(self, init_cash, n_periods, position_directions):
            super().__init__(init_cash, n_periods, position_directions)

    def update_signal(self, time, temp_, close_all=True):
        """Per-bar signal update: manage entries, adaptive exits, and price passthrough."""

        for d_num in temp_.direction_index.keys():
            d_n = temp_.direction_index[d_num]

            # --- Entry ---
            if self.signal_data.loc[time, f'{d_num}_entry_signal']:
                entry_id = find_first_(temp_.open_position[:, d_n], False)
                if entry_id != 'failed':
                    temp_.entry_signal[entry_id, d_n] = True
                    temp_.exit_type[entry_id, d_n] = 0
                    temp_.highest_favorable_price[entry_id, d_n] = 0
                    temp_.trail_active[entry_id, d_n] = False
                    temp_.tp_exit_price[entry_id, d_n] = 0
                    if self.initial_sl_atr is not None:
                        atr = self.signal_data.loc[time, 'ATR']
                        entry_px = self.signal_data.loc[time, f'{d_num}_entry_price']
                        if d_num == 'Long':
                            temp_.sl_price[entry_id, d_n] = entry_px - self.initial_sl_atr * atr
                        else:
                            temp_.sl_price[entry_id, d_n] = entry_px + self.initial_sl_atr * atr
                    else:
                        temp_.sl_price[entry_id, d_n] = 0

            # --- Adaptive exits for open positions ---
            open_ids = find_all_(temp_.open_position[:, d_n], True)
            if open_ids != 'failed':
                atr = self.signal_data.loc[time, 'ATR']
                current_price = self.signal_data.loc[time, f'{d_num}_current_price']

                if d_num == 'Long':
                    temp_.highest_favorable_price[open_ids, d_n] = np.maximum(
                        temp_.highest_favorable_price[open_ids, d_n], current_price)
                    profit = temp_.highest_favorable_price[open_ids, d_n] - temp_.trade_entry_price[open_ids, d_n]
                    trail_stop = temp_.highest_favorable_price[open_ids, d_n] - self.trail_distance_atr * atr
                    sl_hit = (temp_.sl_price[open_ids, d_n] > 0) & (
                        temp_.sl_price[open_ids, d_n] >= self.signal_data.loc[time, f'{d_num}_exit_price'])
                else:
                    temp_.highest_favorable_price[open_ids, d_n] = np.where(
                        temp_.highest_favorable_price[open_ids, d_n] == 0, current_price,
                        np.minimum(temp_.highest_favorable_price[open_ids, d_n], current_price))
                    profit = temp_.trade_entry_price[open_ids, d_n] - temp_.highest_favorable_price[open_ids, d_n]
                    trail_stop = temp_.highest_favorable_price[open_ids, d_n] + self.trail_distance_atr * atr
                    sl_hit = (temp_.sl_price[open_ids, d_n] > 0) & (
                        temp_.sl_price[open_ids, d_n] <= self.signal_data.loc[time, f'{d_num}_exit_price'])

                # Trail activation
                newly_active = profit >= self.trail_activation_atr * atr
                temp_.trail_active[open_ids, d_n] = temp_.trail_active[open_ids, d_n] | newly_active

                # Ratchet stop
                if d_num == 'Long':
                    temp_.sl_price[open_ids, d_n] = np.where(
                        temp_.trail_active[open_ids, d_n],
                        np.maximum(trail_stop, temp_.sl_price[open_ids, d_n]),
                        temp_.sl_price[open_ids, d_n])
                else:
                    temp_.sl_price[open_ids, d_n] = np.where(
                        temp_.trail_active[open_ids, d_n],
                        np.minimum(trail_stop, temp_.sl_price[open_ids, d_n]),
                        temp_.sl_price[open_ids, d_n])

                temp_.exit_type[open_ids, d_n] = np.where(sl_hit, -1, temp_.exit_type[open_ids, d_n])

                # ATR take-profit
                if self.tp_atr is not None:
                    if d_num == 'Long':
                        tp_price = temp_.trade_entry_price[open_ids, d_n] + self.tp_atr * atr
                        tp_hit = self.signal_data.loc[time, 'Long_BidHigh'] >= tp_price
                    else:
                        tp_price = temp_.trade_entry_price[open_ids, d_n] - self.tp_atr * atr
                        tp_hit = self.signal_data.loc[time, 'Short_AskLow'] <= tp_price
                    tp_trigger = tp_hit & (temp_.exit_type[open_ids, d_n] == 0)
                    temp_.tp_exit_price[open_ids, d_n] = np.where(tp_trigger, tp_price, temp_.tp_exit_price[open_ids, d_n])
                    temp_.exit_type[open_ids, d_n] = np.where(tp_trigger, 1, temp_.exit_type[open_ids, d_n])

                # Max holding time
                if self.max_holding_bars is not None:
                    held_too_long = temp_.holding_time[open_ids, d_n] >= self.max_holding_bars
                    temp_.exit_type[open_ids, d_n] = np.where(
                        held_too_long & (temp_.exit_type[open_ids, d_n] == 0), -2, temp_.exit_type[open_ids, d_n])

            # --- Signal-based exits ---
            if self.signal_data.loc[time, f'{d_num}_exit_signal']:
                if close_all:
                    exit_ids = find_all_(temp_.open_position[:, d_n], True)
                else:
                    exit_ids = find_first_(temp_.open_position[:, d_n], True)
                if exit_ids != 'failed':
                    temp_.exit_signal[exit_ids, d_n] = True
            else:
                if close_all:
                    exit_ids = find_all_(temp_.open_position[:, d_n], True)
                else:
                    exit_ids = find_first_(temp_.open_position[:, d_n], True)
                if exit_ids != 'failed':
                    temp_.exit_signal[exit_ids, d_n] = np.where(
                        temp_.exit_type[exit_ids, d_n] != 0, True, False)

            # Past trade tracking
            temp_.past_trade[:, d_n] = np.where(
                temp_.exit_signal[:, d_n] & (temp_.trade_unrealized_pnl[:, d_n] > 0),
                temp_.past_trade[:, d_n] + 1,
                np.where(
                    temp_.exit_signal[:, d_n] & (temp_.trade_unrealized_pnl[:, d_n] < 0),
                    temp_.past_trade[:, d_n] - 1,
                    temp_.past_trade[:, d_n]))

            # --- Set prices ---
            temp_.current_price[:, d_n] = self.signal_data.loc[time, f'{d_num}_current_price']
            temp_.entry_price[:, d_n] = self.signal_data.loc[time, f'{d_num}_entry_price']
            temp_.exit_price[:, d_n] = np.where(
                temp_.exit_type[:, d_n] == -1, temp_.sl_price[:, d_n],
                np.where(temp_.exit_type[:, d_n] == 1, temp_.tp_exit_price[:, d_n],
                np.where(temp_.exit_type[:, d_n] == -2,
                         self.signal_data.loc[time, f'{d_num}_current_price'],
                         self.signal_data.loc[time, f'{d_num}_exit_price'])))
            temp_.leverage[:, d_n] = self.signal_data.loc[time, 'Leverage']

        return 'continue backtest'

"""RSI Mean Reversion strategy.

Entry: RSI crosses into oversold territory (Long) or overbought territory (Short),
       confirmed by price being within a Bollinger Band to ensure the move is
       statistically stretched rather than trending.

Exit: RSI returns to neutral zone (exit signal), ATR-based initial stop loss,
      trailing stop with activation, optional ATR take-profit, optional max holding bars.

Filters:
    - Bollinger Band confirmation: price must be beyond band midpoint in entry direction
    - ATR regime: optional minimum ATR filter to avoid dead markets
    - Session hours: optional time-of-day filter
"""

import numpy as np
import pandas as pd

from engine.records import BacktestRecord_temp, BacktestRecord_strat
from strategies.tools import find_first_, find_all_, backtest_prepare


class RSIMeanReversion:

    def __init__(self, dates: tuple, Tickers: str,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 rsi_exit_neutral: float = 50.0,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 timeframe: str = 'M5',
                 direction: str = 'Both',
                 position_directions: list = None,
                 position_weights: list = None,
                 max_DirectionsPosition: list = None,
                 # ATR exits
                 atr_period: int = 14,
                 initial_sl_atr: float = 2.5,
                 trail_activation_atr: float = 1.0,
                 trail_distance_atr: float = 1.5,
                 tp_atr: float = None,
                 max_holding_bars: int = None,
                 # Optional filters
                 min_atr_pct: float = None,
                 trade_start_hour: int = None,
                 trade_end_hour: int = None,
                 leverage: float = 1.0,
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
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_exit_neutral = rsi_exit_neutral
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.direction = direction
        self.position_directions = position_directions
        self.position_weights = position_weights
        self.max_DirectionsPosition = max_DirectionsPosition
        self.atr_period = atr_period
        self.initial_sl_atr = initial_sl_atr
        self.trail_activation_atr = trail_activation_atr
        self.trail_distance_atr = trail_distance_atr
        self.tp_atr = tp_atr
        self.max_holding_bars = max_holding_bars
        self.min_atr_pct = min_atr_pct
        self.trade_start_hour = trade_start_hour
        self.trade_end_hour = trade_end_hour
        self.leverage = leverage
        self.data_source = kwargs.pop('data_source', None)

        # --- Data pipeline (mirrors MACross exactly) ---
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

        # --- Price mappings ---
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

        # --- Mid prices ---
        mid_close = (data['Long_AskClose'] + data['Long_BidClose']) / 2
        mid_high = (data['Long_AskHigh'] + data['Long_BidHigh']) / 2
        mid_low = (data['Long_AskLow'] + data['Long_BidLow']) / 2

        # --- RSI ---
        delta = mid_close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data['RSI'] = 100 - (100 / (1 + rs))

        # --- Bollinger Bands ---
        bb_mid = mid_close.rolling(window=self.bb_period).mean()
        bb_std = mid_close.rolling(window=self.bb_period).std()
        data['BB_upper'] = bb_mid + self.bb_std * bb_std
        data['BB_lower'] = bb_mid - self.bb_std * bb_std
        data['BB_mid'] = bb_mid

        # --- ATR ---
        prev_close = mid_close.shift(1)
        tr = pd.concat([mid_high - mid_low, (mid_high - prev_close).abs(), (mid_low - prev_close).abs()], axis=1).max(axis=1)
        data['ATR'] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # --- Entry conditions ---
        # Long: RSI crosses up through oversold AND price below BB midpoint (stretched down)
        rsi_oversold_cross = (data['RSI'] >= self.rsi_oversold) & (data['RSI'].shift(1) < self.rsi_oversold)
        price_below_mid = mid_close < data['BB_mid']

        # Short: RSI crosses down through overbought AND price above BB midpoint (stretched up)
        rsi_overbought_cross = (data['RSI'] <= self.rsi_overbought) & (data['RSI'].shift(1) > self.rsi_overbought)
        price_above_mid = mid_close > data['BB_mid']

        long_signal = rsi_oversold_cross & price_below_mid
        short_signal = rsi_overbought_cross & price_above_mid

        # --- Exit conditions ---
        # Exit Long when RSI recovers above neutral; exit Short when RSI drops below neutral
        data['Long_exit_signal'] = data['RSI'] >= self.rsi_exit_neutral
        data['Short_exit_signal'] = data['RSI'] <= (100 - self.rsi_exit_neutral)

        # --- ATR regime filter ---
        if self.min_atr_pct is not None:
            atr_sufficient = (data['ATR'] / mid_close) >= self.min_atr_pct
            long_signal = long_signal & atr_sufficient
            short_signal = short_signal & atr_sufficient

        # --- Session filter ---
        if self.trade_start_hour is not None and self.trade_end_hour is not None:
            hour = data.index.hour
            in_session = (hour >= self.trade_start_hour) & (hour < self.trade_end_hour)
            long_signal = long_signal & in_session
            short_signal = short_signal & in_session

        # --- Direction filter ---
        if self.direction == 'Long':
            short_signal = pd.Series(False, index=data.index)
        elif self.direction == 'Short':
            long_signal = pd.Series(False, index=data.index)

        # Delay 1 bar — enter on next bar's open
        data['Long_entry_signal'] = long_signal.shift(1).fillna(False)
        data['Short_entry_signal'] = short_signal.shift(1).fillna(False)

        self.backtest_data = backtest_prepare(data, self.dates[0])

        self.signal_data = self.backtest_data[[
            'Long_entry_signal', 'Long_exit_signal', 'Long_entry_price', 'Long_current_price', 'Long_exit_price',
            'Short_entry_signal', 'Short_exit_signal', 'Short_entry_price', 'Short_current_price', 'Short_exit_price',
            'Leverage', 'RSI', 'ATR', 'Long_BidHigh', 'Short_AskLow']]

        return self.backtest_data

    class temp_matrices(BacktestRecord_temp):
        def __init__(self, position_directions=['Long', 'Short'], position_weights=[1,1], max_DirectionsPosition=[1,1]):
            super().__init__(position_directions, position_weights, max_DirectionsPosition)
            rows = max(max_DirectionsPosition)
            cols = len(max_DirectionsPosition)
            self.sl_price = np.zeros((rows, cols))
            self.highest_favorable_price = np.zeros((rows, cols))
            self.trail_active = np.full((rows, cols), False, dtype=bool)
            self.tp_exit_price = np.zeros((rows, cols))

    class strat_record(BacktestRecord_strat):
        def __init__(self, init_cash, n_periods, position_directions):
            super().__init__(init_cash, n_periods, position_directions)

    def update_signal(self, bar_idx, temp_, close_all=True):
        """Per-bar signal update: RSI mean reversion entries with ATR trailing stops."""
        sig = self.signal_array
        si = self.signal_index

        for d_num in temp_.direction_index.keys():
            d_n = temp_.direction_index[d_num]

            # --- Entry ---
            if sig[bar_idx, si[f'{d_num}_entry_signal']]:
                entry_id = find_first_(temp_.open_position[:, d_n], False)
                if entry_id != 'failed':
                    temp_.entry_signal[entry_id, d_n] = True
                    temp_.exit_type[entry_id, d_n] = 0
                    temp_.highest_favorable_price[entry_id, d_n] = 0
                    temp_.trail_active[entry_id, d_n] = False
                    temp_.tp_exit_price[entry_id, d_n] = 0

                    atr = sig[bar_idx, si['ATR']]
                    entry_px = sig[bar_idx, si[f'{d_num}_entry_price']]
                    if d_num == 'Long':
                        temp_.sl_price[entry_id, d_n] = entry_px - self.initial_sl_atr * atr
                    else:
                        temp_.sl_price[entry_id, d_n] = entry_px + self.initial_sl_atr * atr

            # --- Adaptive exits for open positions ---
            open_ids = find_all_(temp_.open_position[:, d_n], True)
            if open_ids != 'failed':
                atr = sig[bar_idx, si['ATR']]
                current_price = sig[bar_idx, si[f'{d_num}_current_price']]

                if d_num == 'Long':
                    temp_.highest_favorable_price[open_ids, d_n] = np.maximum(
                        temp_.highest_favorable_price[open_ids, d_n], current_price)
                    profit = (temp_.highest_favorable_price[open_ids, d_n]
                              - temp_.trade_entry_price[open_ids, d_n])
                    trail_stop = (temp_.highest_favorable_price[open_ids, d_n]
                                  - self.trail_distance_atr * atr)
                    sl_hit = (temp_.sl_price[open_ids, d_n] > 0) & (
                        temp_.sl_price[open_ids, d_n] >= sig[bar_idx, si[f'{d_num}_exit_price']])
                else:
                    temp_.highest_favorable_price[open_ids, d_n] = np.where(
                        temp_.highest_favorable_price[open_ids, d_n] == 0, current_price,
                        np.minimum(temp_.highest_favorable_price[open_ids, d_n], current_price))
                    profit = (temp_.trade_entry_price[open_ids, d_n]
                              - temp_.highest_favorable_price[open_ids, d_n])
                    trail_stop = (temp_.highest_favorable_price[open_ids, d_n]
                                  + self.trail_distance_atr * atr)
                    sl_hit = (temp_.sl_price[open_ids, d_n] > 0) & (
                        temp_.sl_price[open_ids, d_n] <= sig[bar_idx, si[f'{d_num}_exit_price']])

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

                temp_.exit_type[open_ids, d_n] = np.where(
                    sl_hit, -1, temp_.exit_type[open_ids, d_n])

                # ATR take-profit
                if self.tp_atr is not None:
                    if d_num == 'Long':
                        tp_price = temp_.trade_entry_price[open_ids, d_n] + self.tp_atr * atr
                        tp_hit = sig[bar_idx, si['Long_BidHigh']] >= tp_price
                    else:
                        tp_price = temp_.trade_entry_price[open_ids, d_n] - self.tp_atr * atr
                        tp_hit = sig[bar_idx, si['Short_AskLow']] <= tp_price
                    tp_trigger = tp_hit & (temp_.exit_type[open_ids, d_n] == 0)
                    temp_.tp_exit_price[open_ids, d_n] = np.where(
                        tp_trigger, tp_price, temp_.tp_exit_price[open_ids, d_n])
                    temp_.exit_type[open_ids, d_n] = np.where(
                        tp_trigger, 1, temp_.exit_type[open_ids, d_n])

                # Max holding time
                if self.max_holding_bars is not None:
                    held_too_long = temp_.holding_time[open_ids, d_n] >= self.max_holding_bars
                    temp_.exit_type[open_ids, d_n] = np.where(
                        held_too_long & (temp_.exit_type[open_ids, d_n] == 0),
                        -2, temp_.exit_type[open_ids, d_n])

            # --- Signal-based exits (RSI neutral recovery) ---
            if sig[bar_idx, si[f'{d_num}_exit_signal']]:
                exit_ids = (find_all_ if close_all else find_first_)(
                    temp_.open_position[:, d_n], True)
                if exit_ids != 'failed':
                    temp_.exit_signal[exit_ids, d_n] = True
            else:
                exit_ids = (find_all_ if close_all else find_first_)(
                    temp_.open_position[:, d_n], True)
                if exit_ids != 'failed':
                    temp_.exit_signal[exit_ids, d_n] = np.where(
                        temp_.exit_type[exit_ids, d_n] != 0, True, False)

            # --- Set prices ---
            temp_.current_price[:, d_n] = sig[bar_idx, si[f'{d_num}_current_price']]
            temp_.entry_price[:, d_n] = sig[bar_idx, si[f'{d_num}_entry_price']]
            temp_.exit_price[:, d_n] = np.where(
                temp_.exit_type[:, d_n] == -1, temp_.sl_price[:, d_n],
                np.where(temp_.exit_type[:, d_n] == 1, temp_.tp_exit_price[:, d_n],
                np.where(temp_.exit_type[:, d_n] == -2,
                         sig[bar_idx, si[f'{d_num}_current_price']],
                         sig[bar_idx, si[f'{d_num}_exit_price']])))
            temp_.leverage[:, d_n] = sig[bar_idx, si['Leverage']]

        return 'continue backtest'
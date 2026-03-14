"""Directional Change (DC) strategy.

Identifies trend reversals using a fixed percentage threshold.
Entry when cumulative price change from a reference point exceeds the threshold;
exit on take-profit (theta), time-based (2t), or DC-based stop loss.

Reference: Directional Change framework (Tsang et al.)
"""

import numpy as np
import pandas as pd

from engine.records import BacktestRecord_temp, BacktestRecord_strat
from strategies.tools import backtest_prepare


class Directional_Change2:

    def __init__(self, dates: tuple, Tickers: str,
                 threshold=0.01, threshold_coef=1,
                 threshold_longCoef=1, threshold_shortCoef=1,
                 timeframe: str = 'H1', stop_PnL: bool = False,
                 direction: str = 'Both',
                 position_directions: list = None,
                 position_weights: list = None,
                 max_DirectionsPosition: list = None,
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
        self.threshold = threshold
        self.threshold_coef = threshold_coef
        self.threshold_longCoef = threshold_longCoef
        self.threshold_shortCoef = threshold_shortCoef
        self.direction = direction
        self.position_directions = position_directions
        self.position_weights = position_weights
        self.max_DirectionsPosition = [1, 1]
        self.last_confirmed_trend = 0
        self.next_confirmed_trend = 0
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

        data['Long_entry_signal'] = False
        data['Long_entry_price'] = data['Long_AskClose']
        data['Long_current_price'] = data['Long_BidClose']
        data['Long_exit_signal'] = False
        data['Long_exit_price'] = data['Long_BidClose']
        data['Long_High'] = data['Long_AskHigh']

        data['Short_entry_signal'] = False
        data['Short_entry_price'] = data['Short_BidClose']
        data['Short_current_price'] = data['Short_AskClose']
        data['Short_exit_signal'] = False
        data['Short_exit_price'] = data['Short_AskClose']
        data['Short_Low'] = data['Short_BidLow']

        data['Leverage'] = self.threshold_coef
        data['Close'] = (data['Long_BidClose'] + data['Long_AskClose']) / 2
        data['Price_change'] = (data['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        data['spread'] = ((data['Long_AskClose'] - data['Long_BidClose']) / data['Long_BidClose']).shift(1)
        data['average_price_change'] = (
            data['Long_AskHigh'].shift(1).rolling(3, min_periods=3).max()
            - data['Long_BidLow'].shift(1).rolling(3, min_periods=3).min()
        ) / data['Long_BidLow'].shift(3)

        self.backtest_data = backtest_prepare(data, self.dates[0])

        self.signal_data = self.backtest_data[[
            'Long_entry_signal', 'Long_exit_signal', 'Long_entry_price',
            'Long_current_price', 'Long_exit_price',
            'Short_entry_signal', 'Short_exit_signal', 'Short_entry_price',
            'Short_current_price', 'Short_exit_price',
            'Price_change', 'Leverage', 'Long_High', 'Short_Low',
            'spread', 'average_price_change', 'Long_BidLow', 'Short_AskHigh'
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
            self.extreme = np.zeros((rows, cols))
            self.eDC_sOS = np.zeros((rows, cols))
            self.DCtime = np.zeros((rows, cols))
            self.stop_loss_seq = np.zeros((rows, cols))

    class strat_record(BacktestRecord_strat):
        def __init__(self, init_cash, n_periods, position_directions):
            super().__init__(init_cash, n_periods, position_directions)

    def update_signal(self, time, temp_, close_all=True):
        """Per-bar signal update using Directional Change logic.

        Tracks trend via price change from reference point. Entry on DC confirmation,
        exit on theta take-profit, 2t time exit, or DC stop loss reversal.
        """
        if time < pd.to_datetime(self.dates[0]):
            return 'continue backtest'

        L = temp_.direction_index['Long']
        S = temp_.direction_index['Short']

        # --- Initial stage: first trend confirming ---
        if self.last_confirmed_trend == 0:
            temp_.DCtime[:, :] += 1

            if self.signal_data.loc[time, 'Price_change'] >= self.threshold:
                self.last_confirmed_trend = 1
                temp_.extreme[:, L] = self.signal_data.loc[time, 'Long_entry_price']
                temp_.eDC_sOS[:, L] = self.signal_data.loc[time, 'Long_entry_price']
                temp_.entry_signal[:, L] = True
                temp_.DCtime[:, S] = 0
            elif self.signal_data.loc[time, 'Price_change'] <= -self.threshold:
                self.last_confirmed_trend = -1
                temp_.extreme[:, S] = self.signal_data.loc[time, 'Short_entry_price']
                temp_.eDC_sOS[:, S] = self.signal_data.loc[time, 'Short_entry_price']
                temp_.entry_signal[:, S] = True
                temp_.DCtime[:, L] = 0

        # --- Active trend: check for exit conditions ---
        elif self.last_confirmed_trend != 0 and self.next_confirmed_trend == 0:
            if self.last_confirmed_trend == 1:
                self._update_long_trend(time, temp_, L, S)
            elif self.last_confirmed_trend == -1:
                self._update_short_trend(time, temp_, L, S)

        # --- Waiting for next DC confirmation after exit ---
        elif self.next_confirmed_trend == -1:
            self._wait_for_short_dc(time, temp_, L, S)
        elif self.next_confirmed_trend == 1:
            self._wait_for_long_dc(time, temp_, L, S)

        # --- Always update prices ---
        temp_.current_price[:, L] = self.signal_data.loc[time, 'Long_current_price']
        temp_.entry_price[:, L] = self.signal_data.loc[time, 'Long_entry_price']
        temp_.exit_price[:, L] = np.where(
            temp_.exit_type[:, L] == 2,
            temp_.stop_loss_price[:, L],
            self.signal_data.loc[time, 'Long_exit_price'])

        temp_.current_price[:, S] = self.signal_data.loc[time, 'Short_current_price']
        temp_.entry_price[:, S] = self.signal_data.loc[time, 'Short_entry_price']
        temp_.exit_price[:, S] = np.where(
            temp_.exit_type[:, S] == 2,
            temp_.stop_loss_price[:, S],
            self.signal_data.loc[time, 'Short_exit_price'])

        temp_.leverage[:, :] = self.threshold_coef
        return 'continue backtest'

    def _update_long_trend(self, time, temp_, L, S):
        """Track upward trend: update extreme, check TP/SL."""
        temp_.extreme[:, L] = np.maximum(
            temp_.extreme[:, L], self.signal_data.loc[time, 'Long_High'])
        temp_.DCtime[:, S] = np.where(
            temp_.extreme[:, L] == self.signal_data.loc[time, 'Long_High'],
            0, temp_.DCtime[:, S] + 1)

        pct_from_entry = (temp_.extreme[:, L] - temp_.eDC_sOS[:, L]) / temp_.eDC_sOS[:, L]
        pct_from_extreme = (self.signal_data.loc[time, 'Long_current_price'] - temp_.extreme[:, L]) / temp_.extreme[:, L]

        if pct_from_entry >= self.threshold:
            temp_.exit_signal[:, L] = True
            temp_.exit_type[:, L] = 1
            self.next_confirmed_trend = -1
        elif temp_.holding_time[:, L] >= temp_.DCtime[:, L] * 2:
            temp_.exit_signal[:, L] = True
            temp_.exit_type[:, L] = 1
            self.next_confirmed_trend = -1
        elif pct_from_extreme <= -self.threshold * 0.5:
            self.last_confirmed_trend = -1
            temp_.eDC_sOS[:, S] = self.signal_data.loc[time, 'Short_entry_price']
            temp_.extreme[:, S] = self.signal_data.loc[time, 'Short_entry_price']
            temp_.exit_signal[:, L] = True
            temp_.exit_type[:, L] = 1
            temp_.entry_signal[:, S] = True

        # Trailing stop activation
        if (temp_.holding_time[:, L] >= temp_.DCtime[:, L]) and not temp_.stop_loss[:, L]:
            if self.signal_data.loc[time, 'Long_current_price'] > temp_.trade_entry_price[:, L]:
                temp_.stop_loss[:, L] = True
                temp_.stop_loss_price[:, L] = (
                    temp_.trade_entry_price[:, L] + self.signal_data.loc[time, 'Long_current_price']) / 2

        if temp_.stop_loss[:, L] and (temp_.stop_loss_price[:, L] >= self.signal_data.loc[time, 'Long_BidLow']):
            temp_.exit_signal[:, L] = True
            temp_.exit_type[:, L] = 2
            self.next_confirmed_trend = -1

    def _update_short_trend(self, time, temp_, L, S):
        """Track downward trend: update extreme, check TP/SL."""
        temp_.extreme[:, S] = np.minimum(
            temp_.extreme[:, S], self.signal_data.loc[time, 'Short_Low'])
        temp_.DCtime[:, L] = np.where(
            temp_.extreme[:, S] == self.signal_data.loc[time, 'Short_Low'],
            0, temp_.DCtime[:, L] + 1)

        pct_from_entry = (self.signal_data.loc[time, 'Short_current_price'] - temp_.eDC_sOS[:, S]) / temp_.eDC_sOS[:, S]
        pct_from_extreme = (self.signal_data.loc[time, 'Short_current_price'] - temp_.extreme[:, S]) / temp_.extreme[:, S]

        if pct_from_entry <= -self.threshold:
            temp_.exit_signal[:, S] = True
            temp_.exit_type[:, S] = 1
            self.next_confirmed_trend = 1
        elif temp_.holding_time[:, S] >= temp_.DCtime[:, S] * 2:
            temp_.exit_signal[:, S] = True
            temp_.exit_type[:, S] = 1
            self.next_confirmed_trend = 1
        elif pct_from_extreme >= self.threshold * 0.5:
            self.last_confirmed_trend = 1
            temp_.eDC_sOS[:, L] = self.signal_data.loc[time, 'Long_entry_price']
            temp_.extreme[:, L] = self.signal_data.loc[time, 'Long_entry_price']
            temp_.exit_signal[:, S] = True
            temp_.exit_type[:, S] = 1
            temp_.entry_signal[:, L] = True

        if (temp_.holding_time[:, S] >= temp_.DCtime[:, S]) and not temp_.stop_loss[:, S]:
            if self.signal_data.loc[time, 'Short_current_price'] < temp_.trade_entry_price[:, S]:
                temp_.stop_loss[:, S] = True
                temp_.stop_loss_price[:, S] = (
                    temp_.trade_entry_price[:, S] + self.signal_data.loc[time, 'Short_current_price']) / 2

        if temp_.stop_loss[:, S] and (temp_.stop_loss_price[:, S] <= self.signal_data.loc[time, 'Short_AskHigh']):
            temp_.exit_signal[:, S] = True
            temp_.exit_type[:, S] = 2
            self.next_confirmed_trend = 1

    def _wait_for_short_dc(self, time, temp_, L, S):
        """After long exit, wait for downward DC to confirm short entry."""
        temp_.extreme[:, L] = np.maximum(
            temp_.extreme[:, L], self.signal_data.loc[time, 'Long_current_price'])
        temp_.DCtime[:, S] = np.where(
            temp_.extreme[:, L] == self.signal_data.loc[time, 'Long_current_price'],
            0, temp_.DCtime[:, S] + 1)

        pct = (self.signal_data.loc[time, 'Long_current_price'] - temp_.extreme[:, L]) / temp_.extreme[:, L]
        if pct <= -self.threshold:
            self.last_confirmed_trend = -1
            self.next_confirmed_trend = 0
            temp_.extreme[:, S] = self.signal_data.loc[time, 'Short_entry_price']
            temp_.eDC_sOS[:, S] = self.signal_data.loc[time, 'Short_entry_price']
            temp_.entry_signal[:, S] = True
            temp_.extreme[:, L] = 0
            temp_.eDC_sOS[:, L] = 0

    def _wait_for_long_dc(self, time, temp_, L, S):
        """After short exit, wait for upward DC to confirm long entry."""
        temp_.extreme[:, S] = np.minimum(
            temp_.extreme[:, S], self.signal_data.loc[time, 'Short_current_price'])
        temp_.DCtime[:, L] = np.where(
            temp_.extreme[:, S] == self.signal_data.loc[time, 'Short_current_price'],
            0, temp_.DCtime[:, L] + 1)

        pct = (self.signal_data.loc[time, 'Short_current_price'] - temp_.extreme[:, S]) / temp_.extreme[:, S]
        if pct >= self.threshold:
            self.last_confirmed_trend = 1
            self.next_confirmed_trend = 0
            temp_.extreme[:, L] = self.signal_data.loc[time, 'Long_entry_price']
            temp_.eDC_sOS[:, L] = self.signal_data.loc[time, 'Long_entry_price']
            temp_.entry_signal[:, L] = True
            temp_.extreme[:, S] = 0
            temp_.eDC_sOS[:, S] = 0

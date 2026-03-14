import numpy as np
from engine.position_manager import PositionManager, MarginCalculator


class BacktestContext:
    """Container for all state needed to backtest a single strategy."""

    def __init__(self, task):
        self.strategy_id = task[3]
        self.strategy_obj = task[4]
        self.temp_dict = task[5]
        self.strat_record = task[6]
        self.shared_principal = task[1]
        self.limit_type = task[2]
        self.timestamp_index = task[0]
        self.id_index = task[8]
        self.weight = task[7]
        self.slippage_pct = task[9] if len(task) > 9 else 0.0
        self.commission_pct = task[10] if len(task) > 10 else 0.0
        self.delay = task[11] if len(task) > 11 else 0
        self.current_idx = 0

    def requeue(self):
        return [
            self.timestamp_index, self.shared_principal, self.limit_type,
            self.strategy_id, self.strategy_obj, self.temp_dict, self.strat_record,
            self.weight, self.id_index,
            self.slippage_pct, self.commission_pct, self.delay
        ]


class StrategyProcessor:
    """Runs the backtest loop for a single strategy across all timestamps.

    Handles signal generation, position management, margin calculation,
    and portfolio record updates. Supports shared-principal mode (multiple
    strategies sharing one capital pool) and signal delay.
    """

    def __init__(self, task, memory_manager):
        self.context = BacktestContext(task)
        self.memory_manager = memory_manager
        self.position_manager = PositionManager()
        self.margin_calculator = MarginCalculator()

        self.signal_delay = self.context.delay
        if self.signal_delay > 0:
            from collections import deque
            self.signal_buffer = deque()

    def run_backtest(self):
        try:
            for timestamp in self.context.timestamp_index.keys():
                result = self._process_timestamp(timestamp)
                if result == 'retask':
                    return 'requeue'
                elif result == 'cash_scenario':
                    continue
            return self.context.temp_dict, self.context.strat_record
        except Exception as e:
            print(f"Error backtesting strategy id@{self.context.strategy_id}: {e}")
            return None

    def _process_timestamp(self, timestamp):
        self.context.current_idx = self.context.timestamp_index[timestamp]

        # Check if this timestamp is ffilled from unified timeline reindexing
        signal_data = self.context.strategy_obj.signal_data
        is_filled = '_is_filled' in signal_data.columns and signal_data.loc[timestamp, '_is_filled']

        if is_filled:
            self._passthrough_filled_prices(timestamp)
        else:
            self.context.strategy_obj.update_signal(
                timestamp, self.context.temp_dict, self.context.temp_dict.close_all)
            self._suppress_untradable_entries(timestamp)

        if self.signal_delay > 0:
            self._apply_signal_delay()

        self.context.temp_dict.position_action = self.position_manager.calculate_postion_action(self.context.temp_dict)

        # Calculate available cash for positions
        if self.context.shared_principal.lower() == 'true':
            retask = self._handle_shared_principal(timestamp)
            if retask:
                return 'retask'
            position_cash = self._shared_principal_position_cash()
        elif self.context.shared_principal.lower() == 'false':
            position_cash = self._calculate_individual_strategy_cash(self.context.current_idx)

        cash_scenario = self._scenario_is_cash(position_cash)
        if cash_scenario:
            return 'cash_scenario'

        deposit = self._process_positions(timestamp, position_cash)
        self._strat_update(deposit)
        self._portfolio_update()

    def _passthrough_filled_prices(self, timestamp):
        """On ffilled bars, only update current prices for open position PnL tracking.
        No signal logic runs - entry/exit signals stay False from cleanup."""
        signal_data = self.context.strategy_obj.signal_data
        temp_ = self.context.temp_dict
        for d in self.context.strategy_obj.position_directions:
            d_idx = temp_.direction_index[d]
            temp_.current_price[:, d_idx] = signal_data.loc[timestamp, f'{d}_current_price']
            temp_.entry_price[:, d_idx] = signal_data.loc[timestamp, f'{d}_entry_price']
            temp_.exit_price[:, d_idx] = signal_data.loc[timestamp, f'{d}_exit_price']

    def _suppress_untradable_entries(self, timestamp):
        """Suppress entry signals on non-tradable bars (ffilled=2, market open false=3, stats prepare=4)."""
        backtest_data = self.context.strategy_obj.backtest_data
        if timestamp not in backtest_data.index:
            return
        temp_ = self.context.temp_dict
        for d in self.context.strategy_obj.position_directions:
            tradable_col = f'{d}_tradable'
            if tradable_col in backtest_data.columns:
                tradable_val = backtest_data.loc[timestamp, tradable_col]
                if tradable_val != 1:
                    d_idx = temp_.direction_index[d]
                    temp_.entry_signal[:, d_idx] = False

    def _handle_shared_principal(self, timestamp):
        try:
            arrays = self.memory_manager.portfolio_arrays
            self.margin_calculator.update_portfolio_margins(arrays, self.context)
            retask = self.margin_calculator._create_retask(self.context, timestamp, arrays['port_begin_balance'])
            return retask
        except Exception as e:
            print(f"Error _handle_shared_principal: {e}")

    def _shared_principal_position_cash(self):
        available_margin = self.margin_calculator.calculate_available_margin(
            self.memory_manager.portfolio_arrays, self.context)
        return available_margin * self.context.weight

    def _calculate_individual_strategy_cash(self, idx):
        strat_record = self.context.strat_record
        if self.context.limit_type.lower() == 'margin':
            return strat_record.free_margin[idx - 1]
        else:
            return strat_record.strat_equity[idx - 1]

    def _scenario_is_cash(self, position_cash):
        temp_ = self.context.temp_dict
        if np.all(temp_.position_action == 'cash') | (
                position_cash <= 0 and np.isin(temp_.position_action, ['cash', 'entry']).all()):
            self._handle_is_cash_scenario()
            self._portfolio_update()
            return True
        return False

    def _handle_is_cash_scenario(self):
        strat_ = self.context.strat_record
        idx = self.context.current_idx
        strat_.strat_action.append('cash')

        if self.context.shared_principal.lower() == 'true':
            strat_.strat_balance[idx] = 0
            strat_.strat_equity[idx] = 0
            strat_.used_margin[idx] = 0
            strat_.free_margin[idx] = 0
            strat_.strat_realized_pnl[idx] = 0
            if idx != 0:
                strat_.strat_positions[idx] = strat_.strat_positions[idx - 1]
                strat_.strat_positions_value[idx] = strat_.strat_positions_value[idx - 1]
                strat_.strat_unrealized_pnl[idx] = strat_.strat_unrealized_pnl[idx - 1]
                strat_.strat_noNewHigh_time[idx] = strat_.strat_noNewHigh_time[idx - 1]
                strat_.strat_keepNewLow_time[idx] = strat_.strat_keepNewLow_time[idx - 1]
                strat_.strat_mdd_amount[idx] = strat_.strat_mdd_amount[idx - 1]
                strat_.strat_mdd[idx] = strat_.strat_mdd[idx - 1]
                strat_.strat_margin_health[idx] = strat_.strat_margin_health[idx - 1]
        elif self.context.shared_principal.lower() == 'false':
            self._strat_filling_next_with_previous(idx)

    def _strat_filling_next_with_previous(self, idx):
        strat_ = self.context.strat_record
        if idx != 0:
            strat_.strat_balance[idx] = strat_.strat_balance[idx - 1]
            strat_.strat_equity[idx] = strat_.strat_equity[idx - 1]
            strat_.used_margin[idx] = strat_.used_margin[idx - 1]
            strat_.free_margin[idx] = strat_.free_margin[idx - 1]
            strat_.strat_positions[idx] = strat_.strat_positions[idx - 1]
            strat_.strat_positions_value[idx] = strat_.strat_positions_value[idx - 1]
            strat_.strat_unrealized_pnl[idx] = strat_.strat_unrealized_pnl[idx - 1]
            strat_.strat_realized_pnl[idx] = strat_.strat_realized_pnl[idx - 1]
            strat_.strat_noNewHigh_time[idx] = strat_.strat_noNewHigh_time[idx - 1]
            strat_.strat_keepNewLow_time[idx] = strat_.strat_keepNewLow_time[idx - 1]
            strat_.strat_mdd_amount[idx] = strat_.strat_mdd_amount[idx - 1]
            strat_.strat_mdd[idx] = strat_.strat_mdd[idx - 1]
            strat_.strat_margin_health[idx] = strat_.strat_margin_health[idx - 1]

    def _strat_update(self, deposit):
        try:
            strat_ = self.context.strat_record
            temp_ = self.context.temp_dict
            idx = self.context.current_idx

            strat_.strat_action.append(temp_.position_action)
            if self.context.shared_principal.lower() == 'true':
                strat_.strat_balance[idx] = deposit
            else:
                strat_.strat_balance[idx] = strat_.strat_balance[idx - 1] + deposit
            strat_.strat_equity[idx] = strat_.strat_balance[idx] + np.sum(temp_.trade_unrealized_pnl)
            strat_.used_margin[idx] = np.sum(temp_.required_margin)
            strat_.free_margin[idx] = strat_.strat_equity[idx] - strat_.used_margin[idx]
            strat_.strat_positions[idx] = np.sum(temp_.open_position, axis=0)
            strat_.strat_positions_value[idx] = np.sum(temp_.current_position_value, axis=0)

            strat_.strat_unrealized_pnl[idx] = np.sum(np.sum(temp_.trade_unrealized_pnl, axis=0))
            strat_.strat_realized_pnl[idx] = deposit

            strat_.strat_hist_high, strat_.strat_noNewHigh = _noNewHigh_time(
                strat_.strat_equity[idx], strat_.strat_hist_high, strat_.strat_noNewHigh)
            strat_.strat_noNewHigh_time[idx] = strat_.strat_noNewHigh

            strat_.strat_hist_low, strat_.strat_keepNewLow = _keepNewLow_time(
                strat_.strat_equity[idx], strat_.strat_hist_low, strat_.strat_keepNewLow)
            strat_.strat_keepNewLow_time[idx] = strat_.strat_keepNewLow

            mdd_amount, mdd_rate = _mdd(strat_.strat_equity[idx], strat_.strat_hist_high)
            strat_.strat_mdd_amount[idx] = mdd_amount
            strat_.strat_mdd[idx] = mdd_rate
            strat_.strat_margin_health[idx] = _margin_health(strat_.strat_equity[idx], strat_.used_margin[idx])

        except Exception as e:
            print(f"Error _strat_update: {e}")

    def _portfolio_update(self):
        try:
            arrays = self.memory_manager.portfolio_arrays
            i = self.context.current_idx
            j = self.context.id_index[self.context.strategy_id]

            arrays['port_end_balance'][i, j] = self.context.strat_record.strat_balance[i]
            arrays['port_equity'][i, j] = self.context.strat_record.strat_equity[i]
            arrays['used_margin'][i, j] = self.context.strat_record.used_margin[i]
        except Exception as e:
            print(f"Error _portfolio_update: {e}")

    def _apply_signal_delay(self):
        temp_ = self.context.temp_dict
        current_entry = temp_.entry_signal.copy()
        current_exit = temp_.exit_signal.copy()
        current_partial_exit = temp_.partial_exit_signal.copy() if hasattr(temp_, 'partial_exit_signal') else None
        current_partial_ratio = temp_.partial_exit_ratio.copy() if hasattr(temp_, 'partial_exit_ratio') else None
        self.signal_buffer.append((current_entry, current_exit, current_partial_exit, current_partial_ratio))

        temp_.open_position = temp_.open_position | current_entry

        temp_.entry_signal[:, :] = False
        temp_.exit_signal[:, :] = False
        if hasattr(temp_, 'partial_exit_signal'):
            temp_.partial_exit_signal[:, :] = False
            temp_.partial_exit_ratio[:, :] = 0.0

        if len(self.signal_buffer) > self.signal_delay:
            delayed_entry, delayed_exit, delayed_partial_exit, delayed_partial_ratio = self.signal_buffer.popleft()
            temp_.entry_signal = delayed_entry
            temp_.exit_signal = delayed_exit
            if delayed_partial_exit is not None and hasattr(temp_, 'partial_exit_signal'):
                temp_.partial_exit_signal = delayed_partial_exit
                temp_.partial_exit_ratio = delayed_partial_ratio

    def _process_positions(self, timestamp, position_cash):
        self.position_manager.cleanup_signal_matrices(self.context.temp_dict)
        self.position_manager.process_opened(self.context.temp_dict)
        self.position_manager.process_entries(
            self.context.temp_dict, timestamp, position_cash, self.context.slippage_pct)
        self.position_manager.process_holdings(self.context.temp_dict)
        deposit = self.position_manager.process_exits(
            self.context.temp_dict, self.context.strat_record, timestamp,
            self.context.slippage_pct, self.context.commission_pct)
        deposit += self.position_manager.process_partial_exits(
            self.context.temp_dict, self.context.strat_record, timestamp,
            self.context.slippage_pct, self.context.commission_pct)
        return deposit


# --- Helper functions (inlined from utils/tools.py) ---

def _mdd(current_value, hist_high):
    drawdown_amount = current_value - hist_high
    drawdown = (drawdown_amount / hist_high) * 100
    return drawdown_amount, drawdown

def _noNewHigh_time(current_value, hist_high, noNewHigh):
    if current_value > hist_high:
        hist_high = current_value
        noNewHigh = 0
    else:
        noNewHigh += 1
    return hist_high, noNewHigh

def _keepNewLow_time(current_value, hist_low, keepNewLow):
    if current_value < hist_low:
        hist_low = current_value
        keepNewLow += 1
    else:
        keepNewLow = 0
    return hist_low, keepNewLow

def _margin_health(equity, used_margin):
    if used_margin == 0:
        return None
    return equity / used_margin * 100

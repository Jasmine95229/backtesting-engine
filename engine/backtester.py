import pandas as pd
import numpy as np
import json
import os
import gc
import inspect
from multiprocessing import Value
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import reduce

from strategies import strategies_object
from engine.records import BacktestRecord_port
from engine.shared_memory import SharedMemoryManager, create_shared_memory
from engine.strategy_processor import StrategyProcessor, _mdd, _noNewHigh_time, _keepNewLow_time, _margin_health
from engine.task_manager import TaskQueue, TaskMemoryManager, BatchCompletionManager

import warnings
warnings.filterwarnings("ignore")


# --- Module-level state for multiprocessing workers ---
temp_port_record = None
pending_queue_size = None

def init_worker(shared_data, queue_size_val):
    global temp_port_record, pending_queue_size
    temp_port_record = shared_data
    pending_queue_size = queue_size_val


def process_strat_backtest(task, n_rows):
    global temp_port_record, pending_queue_size
    memory_manager = SharedMemoryManager(temp_port_record, n_rows)
    processor = StrategyProcessor(task, memory_manager, pending_queue_size)
    result = processor.run_backtest()
    if result == 'requeue':
        return processor.context.requeue()
    return result


def temp_dict_init(cls, available_vars: dict):
    """Instantiate a strategy's temp_matrices class using available config variables."""
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if name in available_vars:
            kwargs[name] = available_vars[name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            raise ValueError(f"Missing required parameter '{name}' and no default value provided.")
    return cls(**kwargs)


def process_strat_record(args):
    """Compile a strategy's backtest arrays into DataFrames for output."""
    timestamp, strat_id, strat_ = args

    df = pd.DataFrame(index=timestamp)
    df['Action'] = strat_.strat_action
    df['Balance'] = strat_.strat_balance
    df['Equity'] = strat_.strat_equity
    df['Used Margin'] = strat_.used_margin
    df['Free Margin'] = strat_.free_margin
    df['Positions'] = strat_.strat_positions.tolist()
    df['Current Position Value'] = strat_.strat_positions_value.tolist()
    df['Unrealized PnL'] = strat_.strat_unrealized_pnl
    df['Realized PnL'] = strat_.strat_realized_pnl
    df['Drawdown Amount'] = strat_.strat_mdd_amount
    df['Maximum Drawdown rate'] = strat_.strat_mdd
    df['Margin Health'] = strat_.strat_margin_health
    df['No NewHigh Time'] = strat_.strat_noNewHigh_time
    df['Keep NewLow Time'] = strat_.strat_keepNewLow_time

    trade_log_cols = [
        'Start Date', 'End Date', 'Position Direction', 'PnL', 'PnL ratio',
        'Holding Time', 'Entry Price', 'Exit Price', 'Leverage', 'Position Size',
        'Exit Type', 'Past Trade', 'Commission', 'Close Type'
    ]
    df_trade_log = pd.DataFrame(
        strat_.trade_log, columns=trade_log_cols,
        index=range(1, len(strat_.trade_log) + 1))

    return df, df_trade_log


def port_record(timestamp, port_, trade_log_dict):
    """Compile portfolio-level results into DataFrames."""
    df = pd.DataFrame(index=timestamp)
    df['Period Begin Balance'] = port_.port_begin_balance
    df['Period End Balance'] = port_.port_end_balance
    df['Equity'] = port_.port_equity
    df['Used Margin'] = port_.used_margin
    df['Free Margin'] = port_.free_margin
    df['Positions'] = port_.port_positions
    df['Unrealized PnL'] = port_.port_unrealized_pnl
    df['Realized PnL'] = port_.port_realized_pnl
    df['Drawdown Amount'] = port_.port_mdd_amount
    df['Maximum Drawdown ratio(%)'] = port_.port_mdd
    df['Margin Health'] = port_.port_margin_health
    df['No NewHigh Time'] = port_.port_noNewHigh_time
    df['Keep NewLow Time'] = port_.port_keepNewLow_time

    for k in trade_log_dict.keys():
        trade_log_dict[k]['Strategy ID'] = k

    trade_log_dfs = list(trade_log_dict.values())
    combined_trade_log = pd.concat(trade_log_dfs, ignore_index=True)
    combined_trade_log = combined_trade_log.sort_values(by='Start Date').reset_index(drop=True)

    return df, combined_trade_log


class portfolioBacktester:
    """Main backtesting orchestrator.

    Loads strategies from a JSON config, constructs a unified timeline,
    runs each strategy through the backtest engine (optionally in parallel
    with shared capital), and compiles results.

    Usage:
        backtester = portfolioBacktester('config/example.json')
        backtester.load_strategies()
        memory_manager = backtester.run_backtest()
        results = backtester.compile_results(memory_manager)
    """

    def __init__(self, config_path: str, dates=('2023-11-20', '2025-04-25')):
        with open(config_path, 'r') as f:
            request_data = json.load(f)

        self.init_cash = request_data['initialCash']
        self.dynamic_weight = request_data['backtestSetting']['dynamic_weight']
        self.shared_principal = request_data['backtestSetting']['shared_principal']
        self.delay = request_data['backtestSetting']['delay']
        self.slippage_pct = request_data['backtestSetting'].get('slippage_pct', 0.0)
        self.commission_pct = request_data['backtestSetting'].get('commission_pct', 0.0)
        self.limit_position_size_type = request_data['backtestSetting']['limit_position_size_type']
        self.start_date = request_data.get('startDate', dates[0])
        self.end_date = request_data.get('endDate', dates[1])
        self.strategies = list(request_data['strategyInfo'])

        self.objs = {}
        self.stats_prepared = {}
        self.signals = {}
        self.weights = {}
        self.temp_dict = {}
        self.strat_dict = {}
        self.strat_timestamp = {}
        self.trade_log = {}

    def load_strategies(self):
        print("Starting strategy loading...")
        batch_size = min(3, len(self.strategies))

        for i in range(0, len(self.strategies), batch_size):
            batch = self.strategies[i:i + batch_size]
            self._process_strategy_batch(batch)
            gc.collect()
            print(f'Processed batch {i // batch_size + 1}/{(len(self.strategies) - 1) // batch_size + 1}')

        self._create_unified_timeline()

        port_n_periods = len(self.portfolio_timestamp)
        self.port_ = BacktestRecord_port(self.init_cash, port_n_periods)
        print(f"Strategy loading completed. Memory usage: {self.port_.get_memory_usage_mb():.1f} MB")

    def _process_strategy_batch(self, batch):
        tasks = [(obj_data, self.start_date, self.end_date, self.init_cash, self.shared_principal)
                 for obj_data in batch]

        with ThreadPoolExecutor(max_workers=min(2, len(batch))) as executor:
            future_to_strat_id = {
                executor.submit(self._load_single_strategy, task): task[0]['id']
                for task in tasks
            }
            for future in as_completed(future_to_strat_id):
                strat_id = future_to_strat_id[future]
                try:
                    result = future.result()
                    if result:
                        self._store_load_strategy(strat_id, result)
                except Exception as e:
                    print(f"Error loading strategy id@{strat_id}: {e}")

    def _load_single_strategy(self, task):
        obj, start_date, end_date, port_init_cash, shared_principal = task
        obj_data = obj.copy()
        strat_type = obj_data.pop('strategy')
        strat_id = obj_data.pop('id')
        strat_w = obj_data.pop('weight')

        try:
            Class = getattr(strategies_object, strat_type)
            instance = Class(**obj_data, dates=(start_date, end_date))
            stats_prepared = instance.prepare_backtest_data()
            signal = instance.signal_data
            temp_ = temp_dict_init(instance.temp_matrices, obj_data)
            weight = strat_w

            if shared_principal.lower() == 'true':
                strat_init_cash = 0
            elif shared_principal.lower() == 'false':
                strat_init_cash = port_init_cash * weight

            n_periods = len(signal.index)
            strat_ = instance.strat_record(strat_init_cash, n_periods, instance.position_directions)

            return {
                "id": strat_id, "instance": instance, "stats": stats_prepared,
                "signal": signal, "weight": weight, "temp": temp_, "strat": strat_
            }
        except TypeError as e:
            print(f"Error _load_single_strategy | strategy@{strat_type} id@{strat_id} |: {e}")
            return None

    def _store_load_strategy(self, strat_id, result):
        self.objs[strat_id] = result["instance"]
        self.stats_prepared[strat_id] = result["stats"]
        self.signals[strat_id] = result["signal"]
        self.weights[strat_id] = result["weight"]
        self.temp_dict[strat_id] = result["temp"]
        self.strat_dict[strat_id] = result["strat"]

    def _create_unified_timeline(self):
        if not self.signals:
            return

        union_index = reduce(
            lambda x, y: x.union(y),
            [df.index for df in self.signals.values() if df is not None])
        self.portfolio_timestamp = union_index.sort_values()

        for strat_id, strat_df in self.signals.items():
            if strat_df is not None:
                df_reindexed = strat_df.reindex(union_index, method='ffill')
                reindexed_mask = ~df_reindexed.index.isin(strat_df.index)

                signal_columns = [col for col in df_reindexed.columns if 'signal' in col]
                for col in signal_columns:
                    df_reindexed.loc[reindexed_mask, col] = False

                df_reindexed['_is_filled'] = reindexed_mask

                self.signals[strat_id] = df_reindexed
                self.objs[strat_id].signal_data = df_reindexed

                strat_init_cash = self.strat_dict[strat_id].strat_balance[0]
                n_periods = len(union_index)
                instance_position_directions = self.objs[strat_id].position_directions
                self.strat_dict[strat_id] = self.objs[strat_id].strat_record(
                    strat_init_cash, n_periods, instance_position_directions)

        # Convert DataFrames to numpy arrays for fast bar-by-bar access
        for strat_id in self.signals:
            obj = self.objs[strat_id]
            obj.signal_array = obj.signal_data.to_numpy()
            obj.signal_index = {col: i for i, col in enumerate(obj.signal_data.columns)}
            obj.backtest_array = obj.backtest_data.to_numpy()
            obj.backtest_col_index = {col: i for i, col in enumerate(obj.backtest_data.columns)}
            obj.backtest_row_index = {ts: i for i, ts in enumerate(obj.backtest_data.index)}

    def run_backtest(self):
        print('Starting backtest...')

        self.timestamp_index = {t: i for i, t in enumerate(self.portfolio_timestamp)}
        self.id_index = {id_: i for i, id_ in enumerate(self.objs.keys())}

        n_rows = len(self.portfolio_timestamp)
        n_cols = len(self.id_index) + 1

        shared_mem = create_shared_memory(n_rows, n_cols)
        memory_manager = SharedMemoryManager(shared_mem, n_rows)

        arrays = memory_manager.portfolio_arrays
        arrays['port_begin_balance'][0] = self.init_cash

        initial_tasks = [
            (self.timestamp_index, self.shared_principal,
             self.limit_position_size_type, strategy_id, self.objs[strategy_id],
             self.temp_dict[strategy_id], self.strat_dict[strategy_id],
             self.weights[strategy_id], self.id_index,
             self.slippage_pct, self.commission_pct, self.delay)
            for strategy_id in self.objs.keys()
        ]

        task_queue = TaskQueue(initial_tasks)
        task_memory_mgr = TaskMemoryManager()
        batch_manager = BatchCompletionManager(len(initial_tasks), batch_size=10)
        futures = {}
        max_workers = max(1, min(3, len(initial_tasks)))

        # Shared counter so workers know if queue has pending work
        queue_size_val = Value('i', task_queue.size())

        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker,
                                 initargs=(shared_mem, queue_size_val)) as executor:
            for _ in range(max_workers):
                task = task_queue.get_task()
                if task:
                    future = executor.submit(process_strat_backtest, task, n_rows)
                    futures[future] = task
                    task_memory_mgr.register_task(future, task)
            queue_size_val.value = task_queue.size()

            while futures:
                for future in as_completed(futures):
                    strat_id = task_memory_mgr.complete_task(future)
                    task = futures.pop(future)

                    try:
                        result = future.result()
                        if result is not None:
                            if len(result) != 2:
                                task_queue.add_task(result)
                                del result
                            else:
                                temp_, strat_ = result
                                self.temp_dict[strat_id] = temp_
                                self.strat_dict[strat_id] = strat_
                                task_queue.mark_completed(strat_id)
                                batch_manager.mark_completed(strat_id)
                                del result, temp_, strat_
                        del task

                        next_task = task_queue.get_task()
                        if next_task:
                            new_future = executor.submit(process_strat_backtest, next_task, n_rows)
                            futures[new_future] = next_task
                            task_memory_mgr.register_task(new_future, next_task)

                        queue_size_val.value = task_queue.size()

                    except Exception as e:
                        print(f'Error backtesting Strategy {strat_id}: {e}')
                        del task
                        next_task = task_queue.get_task()
                        if next_task:
                            new_future = executor.submit(process_strat_backtest, next_task, n_rows)
                            futures[new_future] = next_task
                            task_memory_mgr.register_task(new_future, next_task)
                        queue_size_val.value = task_queue.size()

        batch_manager.force_final_cleanup()
        print('Backtesting completed')
        return memory_manager

    def compile_results(self, memory_manager):
        record_tasks = [(self.portfolio_timestamp, id_, self.strat_dict[id_]) for id_ in self.objs.keys()]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_strat_id = {executor.submit(process_strat_record, task): task[1] for task in record_tasks}
            for future in as_completed(future_to_strat_id):
                strat_id = future_to_strat_id[future]
                try:
                    result = future.result()
                    if result is not None:
                        strat_timestamp, trade_log = result
                        self.strat_timestamp[strat_id] = strat_timestamp
                        self.trade_log[strat_id] = trade_log
                except Exception as e:
                    print(f"Error compile_strat_results {strat_id}: {e}")

        self._calculate_port_matrices(memory_manager)
        return self._general_result

    def _calculate_port_matrices(self, memory_manager):
        timestamp_dfs = list(self.strat_timestamp.values())

        if self.shared_principal.lower() == 'true':
            arrays = memory_manager.portfolio_arrays
            self.port_.port_begin_balance = arrays['port_begin_balance'].tolist()
            self.port_.port_end_balance = arrays['port_end_balance'][:, -1].tolist()
            self.port_.port_equity = arrays['port_equity'][:, -1].tolist()
            self.port_.used_margin = arrays['used_margin'][:, -1].tolist()
            self.port_.free_margin = arrays['free_margin'].tolist()
        else:
            self.port_.port_end_balance = [sum([np.sum(df.loc[i, 'Balance']) for df in timestamp_dfs]) for i in self.portfolio_timestamp]
            self.port_.port_begin_balance[0] = self.init_cash
            self.port_.port_begin_balance[1:] = self.port_.port_end_balance[:-1]
            self.port_.port_equity = [sum([np.sum(df.loc[i, 'Equity']) for df in timestamp_dfs]) for i in self.portfolio_timestamp]
            self.port_.used_margin = [sum([np.sum(df.loc[i, 'Used Margin']) for df in timestamp_dfs]) for i in self.portfolio_timestamp]
            self.port_.free_margin = [sum([np.sum(df.loc[i, 'Free Margin']) for df in timestamp_dfs]) for i in self.portfolio_timestamp]

        self.port_.port_positions = [sum([np.sum(np.maximum(df.loc[i, 'Positions'], 0))for df in timestamp_dfs])for i in self.portfolio_timestamp]
        self.port_.port_unrealized_pnl = [sum([np.sum(df.loc[i, 'Unrealized PnL']) for df in timestamp_dfs]) for i in self.portfolio_timestamp]
        self.port_.port_realized_pnl = [sum([df.loc[i, 'Realized PnL'] for df in timestamp_dfs]) for i in self.portfolio_timestamp]

        for i in range(len(self.port_.port_equity)):
            self.port_.port_hist_high, self.port_.port_noNewHigh = _noNewHigh_time(
                self.port_.port_equity[i], self.port_.port_hist_high, self.port_.port_noNewHigh)
            self.port_.port_noNewHigh_time[i] = self.port_.port_noNewHigh

            self.port_.port_hist_low, self.port_.port_keepNewLow = _keepNewLow_time(
                self.port_.port_equity[i], self.port_.port_hist_low, self.port_.port_keepNewLow)
            self.port_.port_keepNewLow_time[i] = self.port_.port_keepNewLow

            mdd_amount, mdd_rate = _mdd(self.port_.port_equity[i], self.port_.port_hist_high)
            self.port_.port_mdd_amount[i] = mdd_amount
            self.port_.port_mdd[i] = mdd_rate
            self.port_.port_margin_health[i] = _margin_health(self.port_.port_equity[i], self.port_.used_margin[i])

        self.port_timestamp, self.port_trade_log = port_record(self.portfolio_timestamp, self.port_, self.trade_log)

    @property
    def _general_result(self):
        return {
            "port_timestamp": self.port_timestamp,
            "port_trade_log": self.port_trade_log,
            "stats_prepare": self.stats_prepared,
            "strat_signal": self.signals,
            "strat_timestamp": self.strat_timestamp,
            "strat_trade_log": self.trade_log
        }


def save_file(general_results, folder_name, folder_path):
    save_path = os.path.join(folder_path, folder_name)
    os.makedirs(save_path, exist_ok=True)

    for k, v in general_results.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(save_path, f'{k}.csv'), index=True)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                vv.to_csv(os.path.join(save_path, f'{k}_{kk}.csv'), index=True)

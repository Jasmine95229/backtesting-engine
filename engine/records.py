import numpy as np


class BacktestRecord_temp:
    """Temporary per-timestep matrices for tracking open positions.

    Stores entry/exit signals, prices, PnL, and position state as numpy arrays
    with shape (max_positions_per_direction, num_directions).
    """

    def __init__(self, position_directions: list, position_weights: list, max_positions: list, close_all: str = 'True'):
        if close_all.upper() == 'TRUE' or close_all.lower() == 'true':
            self.close_all = True
        else:
            self.close_all = False

        temp_rows = max(max_positions)
        temp_cols = len(max_positions)
        self.direction_index = {d_num: i for i, d_num in enumerate(position_directions)}

        # Signals
        self.open_position = np.full((temp_rows, temp_cols), False, dtype=bool)
        self.entry_signal = np.full((temp_rows, temp_cols), False, dtype=bool)
        self.exit_signal = np.full((temp_rows, temp_cols), False, dtype=bool)
        self.partial_exit_signal = np.full((temp_rows, temp_cols), False, dtype=bool)
        self.partial_exit_ratio = np.zeros((temp_rows, temp_cols))

        # Position tracking
        self.tradable = np.full((temp_rows, temp_cols), False, dtype=bool)
        self.trade_start_date = np.empty((temp_rows, temp_cols), dtype='datetime64[s]')
        self.trade_entry_price = np.zeros((temp_rows, temp_cols))
        self.position_leverage = np.empty((temp_rows, temp_cols))
        self.position_size = np.zeros((temp_rows, temp_cols))
        self.initial_position_value = np.zeros((temp_rows, temp_cols))
        self.trade_current_price = np.zeros((temp_rows, temp_cols))
        self.position_interest = np.zeros((temp_rows, temp_cols))
        self.position_fee = np.zeros((temp_rows, temp_cols))
        self.current_action = np.full((temp_rows, temp_cols), 'empty')
        self.realized_pnl = np.zeros((temp_rows, temp_cols))
        self.required_margin = np.zeros((temp_rows, temp_cols))
        self.entry_price = np.zeros((temp_rows, temp_cols))
        self.exit_price = np.zeros((temp_rows, temp_cols))
        self.current_price = np.zeros((temp_rows, temp_cols))
        self.leverage = np.zeros((temp_rows, temp_cols))
        self.stop_loss = np.full((temp_rows, temp_cols), False, dtype=bool)
        self.stop_loss_price = np.zeros((temp_rows, temp_cols))
        self.trade_exposure = np.zeros((temp_rows, temp_cols))
        self.trade_init_amount = np.zeros((temp_rows, temp_cols))

        # PnL
        self.current_position_value = np.zeros((temp_rows, temp_cols))
        self.trade_unrealized_pnl = np.zeros((temp_rows, temp_cols))
        self.trade_exit_price = np.zeros((temp_rows, temp_cols))
        self.holding_time = np.zeros((temp_rows, temp_cols))
        self.past_trade = np.zeros((temp_rows, temp_cols))
        self.exit_type = np.zeros((temp_rows, temp_cols))

        # Direction multiplier: +1 for Long, -1 for Short
        self.trade_direction = np.ones((temp_rows, temp_cols))
        for i, d in enumerate(position_directions):
            self.trade_direction[:, i] = 1 if 'Long' in d else -1

        self.d_num = np.empty((temp_rows, temp_cols), dtype=object)
        for i, d in enumerate(position_directions):
            self.d_num[:, i] = position_directions[i]

        self.position_weight = np.ones((temp_rows, temp_cols))
        for i, weight in enumerate(position_weights):
            self.position_weight[:, i] = weight


class BacktestRecord_strat:
    """Per-strategy record arrays across the full backtest period.

    Tracks balance, equity, margin, positions, PnL, drawdowns, and trade log.
    """

    def __init__(self, init_cash, n_periods, position_directions):
        self.strat_action = []
        self.strat_balance = np.full(n_periods, np.nan, dtype=np.float32)
        self.strat_equity = np.full(n_periods, np.nan, dtype=np.float32)
        self.used_margin = np.full(n_periods, np.nan, dtype=np.float32)
        self.free_margin = np.full(n_periods, np.nan, dtype=np.float32)

        self.strat_positions = np.full((n_periods, len(position_directions)), np.nan, dtype=np.float16)
        self.strat_positions_value = np.full((n_periods, len(position_directions)), np.nan, dtype=np.float32)
        self.strat_unrealized_pnl = np.full(n_periods, np.nan, dtype=np.float32)
        self.strat_realized_pnl = np.full(n_periods, np.nan, dtype=np.float32)
        self.strat_mdd_amount = np.full(n_periods, np.nan, dtype=np.float32)
        self.strat_mdd = np.full(n_periods, np.nan, dtype=np.float32)
        self.strat_margin_health = np.full(n_periods, np.nan, dtype=np.float32)
        self.strat_noNewHigh_time = np.full(n_periods, 0, dtype=np.int32)
        self.strat_keepNewLow_time = np.full(n_periods, 0, dtype=np.int32)

        self.trade_log = []

        # Initialise first period
        self.strat_balance[0] = init_cash
        self.strat_equity[0] = init_cash
        self.used_margin[0] = 0
        self.free_margin[0] = init_cash
        self.strat_positions[0] = [-1] * len(position_directions)
        self.strat_positions_value[0] = [0] * len(position_directions)
        self.strat_unrealized_pnl[0] = 0
        self.strat_realized_pnl[0] = 0
        self.strat_mdd_amount[0] = 0
        self.strat_mdd[0] = 0
        self.strat_margin_health[0] = 0
        self.strat_hist_high = init_cash
        self.strat_hist_low = init_cash
        self.strat_noNewHigh = 0
        self.strat_keepNewLow = 0


class BacktestRecord_port:
    """Portfolio-level record arrays across the full backtest period."""

    def __init__(self, init_cash, n_periods):
        self.n_periods = n_periods

        self.port_begin_balance = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_end_balance = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_equity = np.full(n_periods, np.nan, dtype=np.float32)
        self.used_margin = np.full(n_periods, np.nan, dtype=np.float32)
        self.free_margin = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_positions = np.full((n_periods, 2), np.nan, dtype=np.float16)
        self.port_positions_values = np.full((n_periods, 2), np.nan, dtype=np.float32)
        self.port_unrealized_pnl = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_realized_pnl = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_mdd_amount = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_mdd = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_margin_health = np.full(n_periods, np.nan, dtype=np.float32)
        self.port_noNewHigh_time = np.full(n_periods, 0, dtype=np.int32)
        self.port_keepNewLow_time = np.full(n_periods, 0, dtype=np.int32)

        self.trade_log = []

        self.port_hist_high = init_cash
        self.port_hist_low = init_cash
        self.port_noNewHigh = 0
        self.port_keepNewLow = 0

    def get_memory_usage_mb(self):
        total_bytes = sum(
            getattr(self, attr).nbytes
            for attr in dir(self)
            if isinstance(getattr(self, attr), np.ndarray)
        )
        return total_bytes / (1024 * 1024)

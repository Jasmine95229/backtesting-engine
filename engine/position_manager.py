import numpy as np


class PositionManager:
    """Handles position lifecycle: entry, hold, exit, partial exit, and cleanup."""

    @staticmethod
    def calculate_postion_action(temp_):
        return np.where(temp_.entry_signal, 'entry',
                   np.where(temp_.exit_signal, 'exit',
                       np.where(temp_.partial_exit_signal & temp_.open_position, 'partial_exit',
                           np.where(temp_.open_position, 'hold', 'cash'))))

    @staticmethod
    def process_opened(temp_):
        open_mask = ~np.isin(temp_.position_action, ['exit', 'cash'])
        temp_.open_position = open_mask

    @staticmethod
    def process_entries(temp_, currentTime, position_open_cash, slippage_pct=0.0):
        entry_mask = temp_.position_action == 'entry'

        temp_.trade_start_date[entry_mask] = currentTime
        temp_.position_leverage = np.where(entry_mask, temp_.leverage, temp_.position_leverage)

        if slippage_pct > 0:
            slipped_price = np.where(
                temp_.trade_direction > 0,
                temp_.entry_price * (1 + slippage_pct),
                temp_.entry_price * (1 - slippage_pct)
            )
            temp_.trade_entry_price = np.where(entry_mask, slipped_price, temp_.trade_entry_price)
        else:
            temp_.trade_entry_price = np.where(entry_mask, temp_.entry_price, temp_.trade_entry_price)

        temp_.trade_init_amount = np.where(entry_mask, position_open_cash * temp_.position_weight, temp_.trade_init_amount)
        temp_.trade_exposure = np.where(entry_mask, temp_.trade_init_amount * temp_.position_leverage, temp_.trade_exposure)

        c = temp_.trade_exposure / temp_.trade_entry_price
        temp_.position_size = np.where(entry_mask, np.round(c, 1), temp_.position_size)
        temp_.initial_position_value = np.where(entry_mask, temp_.trade_entry_price * temp_.position_size, temp_.initial_position_value)
        temp_.required_margin = np.where(entry_mask, temp_.trade_entry_price * temp_.position_size / temp_.position_leverage, temp_.required_margin)

    @staticmethod
    def process_holdings(temp_):
        hold_mask = temp_.position_action == 'hold'
        entry_mask = temp_.position_action == 'entry'
        entry_hold = entry_mask | hold_mask

        temp_.holding_time = np.where(hold_mask, temp_.holding_time + 1, temp_.holding_time)
        temp_.trade_current_price = np.where(entry_hold, temp_.current_price, temp_.trade_current_price)
        temp_.current_position_value = np.where(entry_hold, temp_.trade_current_price * temp_.position_size, temp_.current_position_value)
        temp_.trade_unrealized_pnl = np.where(entry_hold, (temp_.current_position_value - temp_.initial_position_value) * temp_.trade_direction, temp_.trade_unrealized_pnl)

    @staticmethod
    def process_exits(temp_, strat_, currentTime, slippage_pct=0.0, commission_pct=0.0):
        exit_mask = temp_.position_action == 'exit'

        if slippage_pct > 0:
            slipped_exit = np.where(
                temp_.trade_direction > 0,
                temp_.exit_price * (1 - slippage_pct),
                temp_.exit_price * (1 + slippage_pct)
            )
            temp_.trade_exit_price = np.where(exit_mask, slipped_exit, temp_.trade_exit_price)
        else:
            temp_.trade_exit_price = np.where(exit_mask, temp_.exit_price, temp_.trade_exit_price)

        raw_pnl = (temp_.trade_exit_price - temp_.trade_entry_price) * temp_.trade_direction * temp_.position_size

        if commission_pct > 0:
            entry_notional = temp_.trade_entry_price * temp_.position_size
            exit_notional = temp_.trade_exit_price * temp_.position_size
            temp_.position_fee = np.where(exit_mask, (entry_notional + exit_notional) * commission_pct, temp_.position_fee)
        else:
            temp_.position_fee = np.where(exit_mask, 0.0, temp_.position_fee)

        temp_.realized_pnl = np.where(exit_mask, raw_pnl - temp_.position_fee, temp_.realized_pnl)
        deposit = np.sum(temp_.realized_pnl)

        exit_trade = list(zip(*np.where(exit_mask)))
        for trade in exit_trade:
            strat_.trade_log.append([
                temp_.trade_start_date[trade],
                currentTime,
                temp_.d_num[trade],
                temp_.realized_pnl[trade],
                temp_.realized_pnl[trade] / temp_.initial_position_value[trade],
                temp_.holding_time[trade],
                temp_.trade_entry_price[trade],
                temp_.trade_exit_price[trade],
                temp_.position_leverage[trade],
                temp_.position_size[trade],
                temp_.exit_type[trade],
                temp_.past_trade[trade],
                temp_.position_fee[trade],
                'full'
            ])
            PositionManager.cleanup_temp_matrices(temp_, trade)

        return deposit

    @staticmethod
    def process_partial_exits(temp_, strat_, currentTime, slippage_pct=0.0, commission_pct=0.0):
        partial_mask = temp_.position_action == 'partial_exit'
        partial_trades = list(zip(*np.where(partial_mask)))

        deposit = 0.0
        for trade in partial_trades:
            ratio = temp_.partial_exit_ratio[trade]
            if ratio <= 0 or ratio >= 1:
                continue

            exit_price = temp_.exit_price[trade]
            if slippage_pct > 0:
                if temp_.trade_direction[trade] > 0:
                    exit_price = exit_price * (1 - slippage_pct)
                else:
                    exit_price = exit_price * (1 + slippage_pct)

            closed_size = temp_.position_size[trade] * ratio
            entry_price = temp_.trade_entry_price[trade]
            raw_pnl = (exit_price - entry_price) * temp_.trade_direction[trade] * closed_size

            commission = 0.0
            if commission_pct > 0:
                entry_notional = entry_price * closed_size
                exit_notional = exit_price * closed_size
                commission = (entry_notional + exit_notional) * commission_pct

            realized_pnl = raw_pnl - commission

            strat_.trade_log.append([
                temp_.trade_start_date[trade],
                currentTime,
                temp_.d_num[trade],
                realized_pnl,
                realized_pnl / (temp_.initial_position_value[trade] * ratio),
                temp_.holding_time[trade],
                entry_price,
                exit_price,
                temp_.position_leverage[trade],
                closed_size,
                temp_.exit_type[trade],
                temp_.past_trade[trade],
                commission,
                'partial'
            ])

            deposit += realized_pnl

            remaining_ratio = 1.0 - ratio
            temp_.position_size[trade] = round(temp_.position_size[trade] * remaining_ratio, 1)
            temp_.initial_position_value[trade] *= remaining_ratio
            temp_.required_margin[trade] *= remaining_ratio
            temp_.trade_init_amount[trade] *= remaining_ratio
            temp_.trade_exposure[trade] *= remaining_ratio

            temp_.trade_current_price[trade] = temp_.current_price[trade]
            temp_.current_position_value[trade] = temp_.trade_current_price[trade] * temp_.position_size[trade]
            temp_.trade_unrealized_pnl[trade] = (temp_.current_position_value[trade] - temp_.initial_position_value[trade]) * temp_.trade_direction[trade]

        return deposit

    @staticmethod
    def cleanup_signal_matrices(temp_):
        temp_.entry_signal[:, :] = False
        temp_.exit_signal[:, :] = False
        temp_.partial_exit_signal[:, :] = False
        temp_.partial_exit_ratio[:, :] = 0.0

    @staticmethod
    def cleanup_temp_matrices(temp_, trade):
        temp_.trade_start_date[trade] = 0
        temp_.trade_entry_price[trade] = 0
        temp_.position_size[trade] = 0
        temp_.position_leverage[trade] = 0
        temp_.initial_position_value[trade] = 0
        temp_.trade_current_price[trade] = 0
        temp_.current_position_value[trade] = 0
        temp_.trade_unrealized_pnl[trade] = 0
        temp_.realized_pnl[trade] = 0
        temp_.trade_exit_price[trade] = 0
        temp_.holding_time[trade] = 0
        temp_.required_margin[trade] = 0
        temp_.stop_loss[trade] = 0
        temp_.stop_loss_price[trade] = 0
        temp_.position_fee[trade] = 0
        if hasattr(temp_, 'highest_favorable_price'):
            temp_.highest_favorable_price[trade] = 0
        if hasattr(temp_, 'trail_active'):
            temp_.trail_active[trade] = False
        if hasattr(temp_, 'tp_exit_price'):
            temp_.tp_exit_price[trade] = 0
        if hasattr(temp_, 'dca_level'):
            temp_.dca_level[trade] = 0
        if hasattr(temp_, 'slot_type'):
            temp_.slot_type[trade] = 0
        if hasattr(temp_, 'tp_target_price'):
            temp_.tp_target_price[trade] = 0


class MarginCalculator:
    """Calculates available margin and updates portfolio-level shared arrays."""

    @staticmethod
    def calculate_available_margin(arrays, context):
        limit_type = context.limit_type
        if limit_type.lower() == "margin":
            return arrays['free_margin'][context.current_idx - 1]
        elif limit_type.lower() == 'equity':
            return arrays['port_equity'][context.current_idx - 1, -1]
        else:
            return 0

    @staticmethod
    def update_portfolio_margins(arrays, context):
        try:
            max_idx = max(context.timestamp_index.values())
            ready = MarginCalculator._ready_for_update(arrays['port_end_balance'])

            if ready.size > 0:
                total_end_balance = np.nansum(arrays['port_end_balance'][ready, :-1])
                total_equity = np.nansum(arrays['port_equity'][ready, :-1])
                total_used_margin = np.nansum(arrays['used_margin'][ready, :-1])

                arrays['port_end_balance'][ready, -1] = total_end_balance + arrays['port_begin_balance'][ready]
                arrays['port_equity'][ready, -1] = total_equity + arrays['port_begin_balance'][ready]
                arrays['used_margin'][ready, -1] = total_used_margin
                arrays['free_margin'][ready] = arrays['port_equity'][ready, -1] - arrays['used_margin'][ready, -1]

                begin_ready = ready[ready + 1 < max_idx]
                if begin_ready.size > 0:
                    arrays['port_begin_balance'][begin_ready + 1] = arrays['port_end_balance'][begin_ready, -1]

        except Exception as e:
            print(f"Error update_portfolio_margins: {e}")

    @staticmethod
    def _ready_for_update(array):
        (ready,) = np.where(np.all(~np.isnan(array[:, :-1]), axis=1) & np.isnan(array[:, -1]))
        return ready

    @staticmethod
    def _create_retask(context, timestamp, array):
        if np.isin('entry', context.temp_dict.position_action):
            if ~np.isnan(array[context.current_idx]):
                return False
            else:
                context.timestamp_index = {k: v for k, v in context.timestamp_index.items() if k >= timestamp}
                return True
        else:
            return False

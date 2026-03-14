"""Backtesting result visualizations.

Functions:
    bar_monthly              — Monthly PnL bar charts with holding time labels
    plot_cumulative_percentage_change — Equity curve as % change from initial
    statistics_lines         — Overlay indicator lines with optional signal markers
    statistics_event_windows — Per-event zoomed windows around signal timestamps
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Optional, Union


def format_duration(td):
    """Format a timedelta as 'Hh Mm' string."""
    total_seconds = td.total_seconds()
    minutes = total_seconds / 60
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    if total_seconds == 0:
        return "0m"
    elif hours > 0:
        return f"{hours}h {remaining_minutes}m"
    else:
        return f"{remaining_minutes}m"


def bar_monthly(df, fig_name, save_plot=False, save_dir='plots'):
    """Plot monthly PnL bar charts with trade holding time annotations.

    Each subplot shows one calendar month. Bars are colored green (profit)
    or red (loss), with rotated holding-time labels above/below each bar.

    Args:
        df: Trade log DataFrame with 'Start Date', 'End Date', 'PnL' columns.
        fig_name: Title and filename stem for the figure.
        save_plot: If True, saves to {save_dir}/{fig_name}.png.
        save_dir: Directory for saved plots.

    Returns:
        Dict with summary stats: total_pnl, win_num, total_num, win_avg, lose_avg.
    """
    df = df.copy()
    df['Duration'] = df['End Date'] - df['Start Date']
    df['Holding Time String'] = df['Duration'].apply(format_duration)

    df['Trade Month'] = df['Start Date'].dt.to_period('M')
    monthly_groups = df.groupby('Trade Month')
    unique_months = list(monthly_groups.groups.keys())
    num_months = len(unique_months)

    N_COLS = 3
    N_ROWS = (num_months + N_COLS - 1) // N_COLS

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(6 * N_COLS, 5 * N_ROWS), squeeze=False)
    axes = axes.flatten()

    pnl_min = df['PnL'].min()
    pnl_max = df['PnL'].max()
    y_buffer = (pnl_max - pnl_min) * 0.1
    y_min = pnl_min - y_buffer
    y_max = pnl_max + y_buffer

    BAR_WIDTH_DAYS = pd.Timedelta(hours=22).total_seconds() / (24 * 3600)

    for i, month in enumerate(unique_months):
        ax = axes[i]
        data = monthly_groups.get_group(month)

        bars = ax.bar(data['Start Date'], data['PnL'],
                      width=BAR_WIDTH_DAYS,
                      color=np.where(data['PnL'] >= 0, 'g', 'r'))

        for bar, ht_str in zip(bars, data['Holding Time String']):
            yval = bar.get_height()
            text_offset = y_buffer / 20
            if yval >= 0:
                text_y = yval + text_offset
                va = 'bottom'
            else:
                text_y = yval - text_offset
                va = 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, text_y, ht_str,
                    ha='center', va=va, fontsize=6, color='black', rotation=45)

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(month.start_time, month.end_time)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.set_title(month.strftime('%Y-%m'), fontsize=10)
        ax.set_xlabel("Day of Month", fontsize=8)
        ax.set_ylabel("PnL", fontsize=8)
        ax.tick_params(axis='x', rotation=90, labelsize=7)

    for j in range(num_months, N_ROWS * N_COLS):
        fig.delaxes(axes[j])

    fig.suptitle(fig_name, fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4)

    # Summary stats
    win_num = len(df[df['PnL'] > 0])
    total_num = len(df['PnL'])
    win_rate = round(win_num / total_num, 2) if total_num > 0 else 0
    win_avg = round(df.loc[df['PnL'] > 0, 'PnL'].mean(), 2) if win_num > 0 else 0
    lose_avg = round(df.loc[df['PnL'] < 0, 'PnL'].mean(), 2) if (total_num - win_num) > 0 else 0
    total_pnl = round(df['PnL'].sum(), 2)

    text = (f"<Total PnL>: {total_pnl}\n"
            f"<Winning Rate>: {win_num}/{total_num}={win_rate}\n"
            f"<Average Profit>: {win_avg}\n"
            f"<Average Loss>: {lose_avg}")

    fig.text(0.80, 0.96, text, fontsize=10, color='black',
             ha='left', va='center',
             bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'))

    if save_plot:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{fig_name}.png'))
    plt.show()

    return {
        "total_pnl": total_pnl, "win_num": win_num,
        "total_num": total_num, "win_avg": win_avg, "lose_avg": lose_avg
    }


def plot_cumulative_percentage_change(
    dataframes: List[pd.DataFrame],
    column_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Cumulative Percentage Change from Initial Value",
    xlabel: str = "Index",
    ylabel: str = "Cumulative Change (%)",
    x_column: Union[str, None] = None
) -> None:
    """Plot cumulative % change of specified columns relative to their first value.

    Each line shows ((Vt / V0) - 1) * 100 for a given column,
    making it easy to compare performance across different strategies or assets.
    """
    if len(dataframes) != len(column_names):
        raise ValueError("dataframes and column_names must have the same length.")

    plt.figure(figsize=(12, 7))

    for i, df in enumerate(dataframes):
        col_name = column_names[i]
        if col_name not in df.columns:
            continue

        x_values = df.index if x_column is None else df[x_column]
        y_values = df[col_name]
        V0 = y_values.iloc[0]
        cumulative_change = ((y_values / V0) - 1) * 100

        plt.plot(x_values, cumulative_change,
                 label=f"DF {i+1}: {col_name}", marker='.', markersize=4)

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.axhline(0, color='k', linestyle='-', linewidth=1.0, alpha=0.8)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def statistics_lines(
    df: pd.DataFrame,
    cols: List[str],
    save_path: str = "",
    title: str = "Statistics",
    xlabel: str = "Index",
    ylabel: str = "Price",
    signals: bool = False,
    signal_col_name: str = "signal_type",
):
    """Plot indicator lines with optional vertical signal markers.

    Useful for visualizing MA crossovers, ATR bands, or any computed
    columns overlaid on price data.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        ax.plot(df.index, df[col], label=f"Stats {i+1}: {col}", marker=".", markersize=4)

    if signals and signal_col_name in df.columns:
        signal_idx = df.index[df[signal_col_name] != 0]
        signal_vals = df.loc[signal_idx, signal_col_name]
        y_top = ax.get_ylim()[1]

        for x, sig in zip(signal_idx, signal_vals):
            ax.axvline(x=x, linestyle="--", linewidth=1, alpha=0.8)
            ax.text(x, y_top, str(sig), rotation=90, va="top", ha="right", fontsize=8)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def statistics_event_windows(
    df: pd.DataFrame,
    col_names: list,
    signal_col_name: str = "signal_type",
    long_col: str = "Long_AskClose",
    short_col: str = "Short_BidClose",
    long_set=None,
    short_set=None,
    left: int = 10,
    right: int = 10,
    max_events: Optional[int] = None,
    ncols: int = 3,
    save_path: str = "",
    title: str = "Event windows",
    ylabel: str = "Price",
):
    """Plot zoomed windows around each signal event.

    Each subplot shows +-left/right bars around a signal timestamp,
    with the relevant price columns plotted. Long and short signals
    can optionally show different column subsets.
    """
    mask = df[signal_col_name].ne(0)
    event_pos = df.index.get_indexer(df.index[mask])
    event_pos = event_pos[event_pos >= 0]

    if max_events is not None:
        event_pos = event_pos[:max_events]

    n = len(event_pos)
    if n == 0:
        raise ValueError("No non-zero signals found.")

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.ravel()

    legend_map = {}
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {c: default_colors[i % len(default_colors)] for i, c in enumerate(col_names)}

    for k, p in enumerate(event_pos):
        ax = axes[k]
        start = max(0, p - left)
        end = min(len(df), p + right + 1)
        w = df.iloc[start:end]

        sig_raw = df.iloc[p][signal_col_name]
        sig_int = None
        if pd.notna(sig_raw):
            try:
                sig_int = int(float(sig_raw))
            except Exception:
                sig_int = None

        # Select columns based on signal direction
        if long_set is not None and short_set is not None:
            if sig_int in long_set:
                cols_to_plot = [c for c in col_names if 'Short' not in c]
            elif sig_int in short_set:
                cols_to_plot = [c for c in col_names if 'Long' not in c]
            else:
                cols_to_plot = col_names
        else:
            cols_to_plot = col_names

        for col in cols_to_plot:
            if col in w.columns:
                line, = ax.plot(w.index, w[col], marker=".", markersize=3,
                                label=col, color=color_map.get(col))
                legend_map[col] = line

        x_event = df.index[p]
        ax.axvline(x=x_event, linestyle="--", linewidth=1)
        ax.set_title(f"event#{k+1} @ {x_event} | sig={sig_raw}", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)

    if legend_map:
        fig.legend(legend_map.values(), legend_map.keys(),
                   loc="upper left", bbox_to_anchor=(0.01, 0.97),
                   fontsize=9, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

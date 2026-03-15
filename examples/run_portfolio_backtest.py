"""Portfolio backtest example — 3 heterogeneous strategies, shared capital.

This example demonstrates the core architectural feature of this engine:
running strategies with completely different logic (trend-following, momentum
breakout, mean reversion) concurrently via shared-memory multiprocessing,
without duplicating price data across processes.

Strategies
----------
A. MACross       — EMA crossover on NAS100 M5  (trend-following)
B. ATRBreakout   — Price breakout on EURUSD H1  (momentum)
C. RSIMeanReversion — RSI + Bollinger Band on EURUSD M5  (counter-trend)

All three run in parallel via ProcessPoolExecutor, reading from a shared
memory buffer. Capital is pooled (shared_principal = True), allocated by
weight per strategy.

Usage
-----
    python examples/run_portfolio_backtest.py

Data required
-------------
    DB/FXCM/M5/NAS100.csv
    DB/FXCM/H1/EURUSD.csv
    DB/FXCM/M5/EURUSD.csv

    Format: Date,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,Volume,Timestamp
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Add project root to path so imports work from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtester import portfolioBacktester, save_file
from plotting.charts import bar_monthly, plot_cumulative_percentage_change


def main():
    config_path = 'config/example_portfolio.json'
    step_times = {}
    total_start = time.perf_counter()

    # --- 1. Load strategies ---
    print("=" * 60)
    print("Portfolio Backtester")
    print("=" * 60)

    t0 = time.perf_counter()
    backtester = portfolioBacktester(config_path=config_path)
    backtester.load_strategies()
    step_times['1. Load strategies'] = time.perf_counter() - t0

    # --- 2. Run backtest ---
    t0 = time.perf_counter()
    memory_manager = backtester.run_backtest()
    step_times['2. Run backtest'] = time.perf_counter() - t0

    # --- 3. Compile results ---
    t0 = time.perf_counter()
    results = backtester.compile_results(memory_manager)
    step_times['3. Compile results'] = time.perf_counter() - t0

    # --- 4. Print summary ---
    port = results['port_timestamp']
    start = port.index[0]
    end = port.index[-1]
    years = (end - start).total_seconds() / (365 * 24 * 60 * 60)

    final_equity = port['Equity'].iloc[-1]
    initial_equity = port['Equity'].iloc[0]
    period_return = (final_equity / initial_equity) - 1
    annual_return = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0
    max_mdd = pd.Series(port['Maximum Drawdown ratio(%)']).replace('nan', np.nan).dropna().min()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Period:         {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({years:.2f} years)")
    print(f"Final Equity:   {final_equity:.2f}")
    print(f"Period Return:  {period_return:.2%}")
    print(f"Annual Return:  {annual_return:.2%}")
    print(f"Max Drawdown:   {max_mdd:.2f}%")
    print(f"Total Trades:   {len(results['port_trade_log'])}")

    # --- 5. Save results ---
    t0 = time.perf_counter()
    save_file(results, 'Portfolio_example', 'output')
    step_times['5. Save results'] = time.perf_counter() - t0
    print(f"\nResults saved to output/Portfolio_example/")

    # --- 6. Plot ---
    t0 = time.perf_counter()
    bar_monthly(results['port_trade_log'],
                fig_name="Portfolio_example",
                save_plot=True, save_dir='plots')

    plot_cumulative_percentage_change(
        dataframes=[port],
        column_names=['Equity'],
        title="Portfolio Equity Curve",
        save_path='plots/Portfolio_equity.png')
    step_times['6. Plot'] = time.perf_counter() - t0

    print("Plots saved to plots/")

    # --- Timing summary ---
    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print("TIMING")
    print("=" * 60)
    for step, elapsed in step_times.items():
        print(f"  {step:<25s} {elapsed:>8.2f}s")
    print(f"  {'TOTAL':<25s} {total_elapsed:>8.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
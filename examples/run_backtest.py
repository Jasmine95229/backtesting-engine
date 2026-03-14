"""Example: Run a MACross backtest on NAS100 M5 data.

Usage:
    cd backtesting-engine
    python examples/run_backtest.py

Prerequisites:
    - Price data CSV in DB/FXCM/M5/NAS100.csv
      (see README.md for CSV format)
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path so imports work from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtester import portfolioBacktester, save_file
from plotting.charts import bar_monthly, plot_cumulative_percentage_change


def main():
    config_path = 'config/example_macross.json'

    # --- 1. Load strategies ---
    print("=" * 60)
    print("Portfolio Backtester")
    print("=" * 60)

    backtester = portfolioBacktester(config_path=config_path)
    backtester.load_strategies()

    # --- 2. Run backtest ---
    memory_manager = backtester.run_backtest()

    # --- 3. Compile results ---
    results = backtester.compile_results(memory_manager)

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
    save_file(results, 'MACross_example', 'output')
    print(f"\nResults saved to output/MACross_example/")

    # --- 6. Plot ---
    bar_monthly(results['port_trade_log'],
                fig_name="MACross_NAS100_M5",
                save_plot=True, save_dir='plots')

    plot_cumulative_percentage_change(
        dataframes=[port],
        column_names=['Equity'],
        title="Portfolio Equity Curve",
        save_path='plots/MACross_NAS100_M5_equity.png')

    print("Plots saved to plots/")


if __name__ == "__main__":
    main()

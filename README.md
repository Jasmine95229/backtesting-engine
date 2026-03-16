# Backtesting Engine

A Python portfolio backtesting engine for research-grade strategy development across equities, indices, FX, and crypto. Built around three problems that existing frameworks leave unsolved.


---

## Why this exists

Most backtesting frameworks handle a single strategy well. Scaling to a portfolio of heterogeneous strategies — different logic, different instruments, different timeframes — exposes gaps that none of the popular tools fully address.

### Problem 1 — Forward-filled bar bias

When data providers gap low-liquidity periods (weekends, holidays, thin sessions), they forward-fill the previous bar's price. The bar looks normal. Indicators compute on it. Signals fire.

But nothing traded on that bar. The signal is noise injected directly into your backtest.

**Backtrader, VectorBT, and LEAN have no mechanism to detect or suppress these bars.**

This engine labels every bar with a tradability flag during preprocessing and skips signal logic on forward-filled bars entirely — before any indicator or entry logic runs.

### Problem 2 — Heterogeneous parallel execution

VectorBT parallelises parameter variants of the same strategy by broadcasting arrays. Running strategies with fundamentally different logic requires either multiple separate processes (LEAN, Backtrader `optstrategy`) or sequential execution.

This engine runs heterogeneous strategies — different classes, different instruments, different timeframes — concurrently via `ProcessPoolExecutor` with a **shared memory data buffer**. Price data is written once and read by all worker processes without duplication.

### Problem 3 — Weighted capital allocation in shared mode

Backtrader's shared broker is first-come-first-served. One strategy can consume all available cash before others get a chance to trade. There is no native per-strategy weight enforcement.

This engine allocates capital by weight per strategy in shared principal mode. Each strategy's buying power is bounded by its configured weight, regardless of execution order.

---

### Design tradeoff

This engine prioritises **correctness and expressiveness** over raw speed. Per-bar Python logic is slower than fully vectorized engines like VectorBT. The target use case is research workflows where strategy logic is complex, data quality matters, and you need confidence that your backtest results reflect what would actually happen — not artefacts introduced by data gaps or capital contention.

---

## Features

### Core differentiators
- **ffill bar suppression** — forward-filled bars are detected during preprocessing and excluded from signal logic, eliminating a hidden source of look-ahead bias
- **Heterogeneous parallel execution** — strategies with completely different logic run concurrently via `ProcessPoolExecutor` with shared memory; no data duplication across processes
- **Weighted capital allocation** — shared principal mode enforces per-strategy weights, not first-come-first-served

### Portfolio management
- Multi-strategy portfolios with shared or independent capital pools
- Vectorized position matrix — supports concurrent long and short positions per instrument per strategy
- Partial exits — close a fraction of a position and scale down the remainder
- Per-strategy capital weights with JSON config

### Data integrity
- Tradable-bar labeling — rest periods, market-open gaps, ffill bars, and lookback periods are flagged and respected
- Bid/Ask OHLC native support — longs fill on Ask prices, shorts fill on Bid prices
- Signal delay — buffer N bars between signal generation and execution to prevent look-ahead bias
- Multi-timeframe safe — unified timeline ffill detection prevents cross-timeframe signal contamination

### Execution realism
- Slippage modeled as fraction of price, direction-aware (longs pay more on entry, receive less on exit)
- Commission on round-trip notional
- Spin-wait requeue — when no other tasks are queued, shared-principal retasks spin in-process to avoid cross-process pickle overhead

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/backtesting-engine.git
cd backtesting-engine
pip install -r requirements.txt

# 2. Add price data
#    Place CSV files in DB/{source}/{timeframe}/{ticker}.csv
#    Example: DB/FXCM/M5/NAS100.csv
#    Format: Date,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,Volume,Timestamp

# 3. Run the single-strategy example
python examples/run_backtest.py

# 4. Run the multi-strategy portfolio example
python examples/run_portfolio_backtest.py
```

```python
# Use as a library
from engine.backtester import portfolioBacktester, save_file
from plotting.charts import bar_monthly

backtester = portfolioBacktester('config/example_macross.json')
backtester.load_strategies()
memory_manager = backtester.run_backtest()
results = backtester.compile_results(memory_manager)

save_file(results, 'MyBacktest', 'output')
bar_monthly(results['port_trade_log'], fig_name="MyBacktest", save_plot=True)
```

The `examples/` directory contains two runnable scripts:
- `run_backtest.py` — single MACross strategy on NAS100 M5
- `run_portfolio_backtest.py` — three heterogeneous strategies (trend-following, momentum breakout, mean reversion) across two instruments, demonstrating shared memory parallel execution

---

## System Architecture

```
  JSON Config ──> portfolioBacktester (Orchestrator)
                    |
          ┌─────────┼──────────┐
          v         v          v
    Strategy     ProcessPool   Result
    Loader       Executor      Compiler
    (Thread)     (parallel)    (Thread)
          |         |          |
          v         v          v
    Strategy   StrategyProcessor   DataFrames
    Classes    per-bar loop:       + trade logs
               1. update_signal()
               2. suppress ffilled bars  ← ffill guard
               3. signal delay
               4. position actions
               5. entry/hold/exit
               6. portfolio update
                    |
                    v
              PositionManager          SharedMemoryManager
              (vectorized numpy)       (cross-process state)
                    |
                    v
              Spin-wait retask         pending_queue_size == 0 →
              (avoids requeue           spin in-process;
               pickle overhead)         > 0 → requeue normally
```

### Data Flow

```
Historical CSVs (DB/)
      │
      v
data/loader.py              Load CSV, timezone convert (UTC → US/Eastern)
      │
      v
data/time_utils.py          Generate tradable-time grid per instrument type
      │
      v
data/preprocessing.py       Align prices to grid
                            ffill gaps → tradable flag = 2  ← labeled here
                            Prefix columns by direction (Long_*, Short_*)
                            Mark market-open gaps → tradable flag = 3
      │
      v
Strategy.prepare_backtest_data()    Compute indicators & pre-generate signals
      │
      v
_create_unified_timeline()  Merge all strategy timelines, mark _is_filled
      │
      v
StrategyProcessor           Bar-by-bar simulation
                            ffill bars → update_signal() skipped entirely
      │
      v
compile_results()           Build output DataFrames + trade logs
```

---

## Configuration

Backtests are driven by JSON config files. See `config/example_macross.json` for a single-strategy example and `config/example_portfolio.json` for a multi-strategy portfolio.

### Backtest Settings

| Field | Type | Description |
|---|---|---|
| `initialCash` | float | Starting capital |
| `shared_principal` | string | `"True"`: strategies share one capital pool, allocated by `weight`. `"False"`: each strategy gets `initialCash * weight` independently |
| `delay` | int | Signal delay in bars (0 = immediate execution on same bar) |
| `limit_position_size_type` | string | `"equity"` or `"margin"` — basis for available cash calculation |
| `slippage_pct` | float | Slippage as fraction of price (e.g. `0.0001` = 1 bps) |
| `commission_pct` | float | Commission as fraction of round-trip notional |

### Strategy Parameters

Each entry in `strategyInfo` requires:

| Field | Description |
|---|---|
| `strategy` | Class name — must be registered in `strategies/strategies_object.py` |
| `Tickers` | Instrument name matching the CSV filename in `DB/` |
| `timeframe` | Bar period: `"M5"`, `"H1"`, `"1D"`, etc. |
| `leverage` | Position leverage multiplier |
| `position_directions` | Direction labels, e.g. `["Long", "Short"]` — controls position matrix columns |
| `max_DirectionsPosition` | Max concurrent positions per direction, e.g. `[1, 1]` — controls position matrix rows |
| `position_weights` | Capital allocation per direction, e.g. `[1, 1]` |
| `weight` | Portfolio-level capital weight for this strategy |
| `id` | Unique strategy identifier used in trade log output |
| `data_source` | Data provider subfolder in `DB/`, e.g. `"FXCM"` |

Additional strategy-specific parameters (MA periods, ATR multipliers, RSI thresholds, etc.) are passed directly to the strategy constructor via `**kwargs`.

---

## Adding a Custom Strategy

Inherit from `StrategyBase` and implement `prepare_backtest_data()` and `update_signal()`, then register the class in `strategies/strategies_object.py`. See the existing strategies in `strategies/examples/` as reference implementations.

---



## Position Matrix Design

The core state container is `BacktestRecord_temp` — a set of 2D numpy arrays with shape `(max_positions, num_directions)`:

- **Columns** = trading directions (e.g. `["Long", "Short"]`)
- **Rows** = concurrent position slots (e.g. `max_DirectionsPosition = [3, 2]` → 3 Long slots, 2 Short slots)

All position operations (`process_entries`, `process_holdings`, `process_exits`) are vectorized across the full matrix in a single numpy call — no Python loops over individual positions.

```
Example: 1 Long slot + 1 Short slot

              Long    Short
  slot 0  [  True  ,  False  ]    entry_signal
  slot 0  [ 18500  ,   0.0   ]    trade_entry_price
  slot 0  [   1.0  ,   0.0   ]    open_position (bool)
```

---

## Tradable Bar Labels

Every bar is assigned a tradability flag during preprocessing. The engine checks this flag before running any signal logic.

| Value | Meaning | Engine behaviour |
|---|---|---|
| `1` | Normal tradable bar | Full signal logic runs |
| `0` | Rest period or last candle of session | Entry suppressed |
| `2` | **Forward-filled** (data gap) | Entry suppressed |
| `3` | Market-open gap bar | Entry suppressed |
| `4` | Statistics lookback warmup period | Entry suppressed |
| `_is_filled=True` | Unified timeline ffill across strategies | `update_signal()` skipped entirely — only price passthrough |

Labels `2` and `_is_filled=True` are the primary defence against forward-fill bias.

---

## Trading Cost Model

**Slippage** is direction-aware and adjusts fill prices:

| Direction | Entry | Exit |
|---|---|---|
| Long | `price × (1 + slippage_pct)` — pay more | `price × (1 − slippage_pct)` — receive less |
| Short | `price × (1 − slippage_pct)` — receive more | `price × (1 + slippage_pct)` — pay more |

**Commission** on round-trip notional:
```
commission = (entry_notional + exit_notional) × commission_pct
realized_pnl = raw_pnl − commission
```

---



## Project Structure

```
backtesting-engine/
├── engine/
│   ├── backtester.py           portfolioBacktester orchestrator
│   ├── strategy_processor.py   Per-strategy bar-by-bar loop + ffill guard
│   ├── position_manager.py     PositionManager + MarginCalculator
│   ├── records.py              BacktestRecord_temp / _strat / _port
│   ├── shared_memory.py        Cross-process shared arrays
│   └── task_manager.py         TaskQueue + memory cleanup managers
│
├── data/
│   ├── loader.py               CSV reading + timezone conversion
│   ├── preprocessing.py        Price alignment, tradable flags, anomaly check
│   └── time_utils.py           Trading hours masks by instrument type
│
├── strategies/
│   ├── strategies_object.py    Strategy class registry
│   ├── base.py                 StrategyBase — shared data pipeline
│   ├── tools.py                Helper functions (find_first_, find_all_, backtest_prepare)
│   └── examples/
│       ├── ma_cross.py         MACross — EMA crossover with adaptive exits
│       ├── atr_breakout.py     ATRBreakout — channel breakout with ATR filter
│       └── rsi_mean_reversion.py  RSIMeanReversion — RSI + Bollinger Band counter-trend
│
├── plotting/
│   └── charts.py               Monthly PnL bars, equity curves, event windows
│
├── config/
│   ├── example_macross.json    Single-strategy annotated config
│   └── example_portfolio.json  Three-strategy portfolio config
│
├── examples/
│   ├── run_backtest.py         Single-strategy example
│   └── run_portfolio_backtest.py  Multi-strategy portfolio example
│
├── samples/
│   ├── monthly_pnl_example.png
│   └── trade_log_example.csv
│
├── DB/                         Price data — gitignored
│   └── {source}/{timeframe}/{ticker}.csv
├── output/                     Generated results — gitignored
└── plots/                      Generated charts — gitignored
```

---

## Data Source & Trading Hours

### CSV Format

Place price data in `DB/{source}/{timeframe}/{ticker}.csv`:

```csv
Date,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose,Volume,Timestamp
2024-01-02 18:00:00,16800.5,16815.2,16798.1,16810.3,16801.5,16816.2,16799.1,16811.3,1250,1704232800
```

Timestamps are assumed UTC and converted to US/Eastern internally.

---

### ⚠️ Trading Hours Are Calibrated for FXCM Data

The tradable-bar labeling system in `data/time_utils.py` is **calibrated against FXCM's trading schedule**. The engine uses these hardcoded session definitions (all times US/Eastern) to generate the tradable-time grid before aligning your CSV data against it:

| Type | Instruments | Session window | Notes |
|---|---|---|---|
| `type_1` | FX majors, crypto (`EURUSD`, `GBPUSD`, `BTCUSD`, ...) | Sun 17:00 – Fri 16:55 | — |
| `type_2` | Indices, commodities (`NAS100`, `SPX500`, `XAUUSD`, ...) | Sun 18:00 – Fri 16:45 | Hour 17:xx excluded as maintenance window |
| `type_3` | US stocks (ticker suffix `.ext`) | Sun 20:00 – Fri 16:00 | — |

**Unknown tickers default to `type_2`.**

This means:

- If your data comes from a broker whose maintenance window differs from FXCM's (e.g. hour 22:xx instead of 17:xx), bars during that window will be **incorrectly classified as tradable** rather than suppressed.
- If your broker's FX session opens at a different time (e.g. 17:00 vs 18:00 Sunday), the market-open gap detection (`tradable=3`) will misfire.
- Holiday closures are not currently modeled — if your broker closes early on public holidays, those bars will not be flagged.

**The tradable bar system only works correctly when your data source matches the session definitions in `time_utils.py`.**

---

### Adapting to a Different Data Source

To use data from a broker other than FXCM, update `data/time_utils.py`:

**1. Add your instruments to the correct type, or create a new type:**

```python
TRADING_HOURS = {
    'type_1': [
        # Add your FX/crypto tickers here
        'EURUSD', 'GBPUSD', 'USDJPY', ...
    ],
    'type_2': [
        # Add your index/commodity tickers here
        'NAS100', 'SPX500', 'XAUUSD', ...
    ],
}
```

**2. Adjust the session window in the corresponding `_type_N_masks()` function:**

```python
def _type_2_masks(df, tf):
    # Change 18 to your broker's Sunday open hour (US/Eastern)
    fx_mask = (
        ((df['Weekday'] == 6) & (df['Hour'] >= 18)) |  # ← adjust this
        ...
    )
    # Change 17 to your broker's maintenance/closed hour, or remove if none
    mask_not = df['Hour'] == 17  # ← adjust or remove this
```

**3. Update the maintenance window exclusion** (`mask_not` in `_type_2_masks`) to match your broker's daily maintenance schedule, or set it to an empty mask if your broker has no maintenance window:

```python
mask_not = pd.Series(False, index=df.index)  # no maintenance window
```

If you are using data from **Interactive Brokers, Dukascopy, Pepperstone, or another provider**, verify their session boundaries and maintenance windows before running a backtest. Incorrect session definitions will silently produce wrong tradable labels — bars that should be suppressed will generate signals, and the ffill bias protection will be partially ineffective.

---

## Trade Log Columns

| Column | Description |
|---|---|
| `Start Date` | Entry timestamp |
| `End Date` | Exit timestamp |
| `Position Direction` | `"Long"` or `"Short"` |
| `PnL` | Realized profit/loss after commission |
| `PnL ratio` | PnL / initial position value |
| `Holding Time` | Bars held |
| `Entry Price` | Fill price at entry (after slippage) |
| `Exit Price` | Fill price at exit (after slippage) |
| `Leverage` | Position leverage |
| `Position Size` | Units |
| `Exit Type` | Strategy-defined exit reason code |
| `Past Trade` | Strategy-defined trade metadata |
| `Commission` | Total commission charged |
| `Close Type` | `"full"` or `"partial"` |

---

## Dependencies

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24
- matplotlib >= 3.7

---

## License

MIT
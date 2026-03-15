"""Strategy registry — maps strategy class names to implementations.

The backtester resolves strategy classes by name via getattr(strategies_object, name).
Import all available strategies here so they're discoverable.
"""

from strategies.examples.ma_cross import MACross
from strategies.examples.atr_breakout import ATRBreakout
from strategies.examples.rsi_mean_reversion import RSIMeanReversion

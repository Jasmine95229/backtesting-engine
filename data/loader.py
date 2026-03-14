import os
import pandas as pd
from pathlib import Path


# Default database path relative to project root
DB_PATH = Path(__file__).resolve().parent.parent / "DB"


def TimeZone_convert(ori_time, from_zone: str = 'UTC', to_zone: str = 'US/Eastern'):
    """Convert a DatetimeIndex from one timezone to another, returning a naive (tz-unaware) result.

    Tradable time labels are stored in US/Eastern without tz info to keep
    downstream pandas operations simple (no mixed-tz alignment issues).
    """
    if ori_time.tz is None:
        converted_time = ori_time.tz_localize(from_zone)
    else:
        converted_time = ori_time

    converted_time = converted_time.tz_convert(to_zone)
    naive_time = converted_time.tz_localize(None)

    return naive_time


def histPrices(ticker, timeframe='H1', source=None, db_path=None):
    """Load historical OHLC price data from CSV.

    CSV format: Date, BidOpen, BidHigh, BidLow, BidClose,
                AskOpen, AskHigh, AskLow, AskClose, Volume, Timestamp

    Args:
        ticker: Instrument name (e.g. 'NAS100', 'BTCUSD').
        timeframe: Bar period (e.g. 'M5', 'H1', '1D').
        source: Data provider subfolder (e.g. 'FXCM', 'yfinance').
                If None, uses legacy flat path DB/{timeframe}/.
        db_path: Override for the database root directory.

    Returns:
        pd.DataFrame indexed by Date (US/Eastern, tz-naive), or None if file not found.
    """
    base = Path(db_path) if db_path else DB_PATH

    if source:
        hist_path = base / source / timeframe / f'{ticker.replace("/", "")}.csv'
    else:
        hist_path = base / timeframe / f'{ticker.replace("/", "")}.csv'

    if not os.path.exists(hist_path):
        print(f"{ticker} data not found at {hist_path}")
        return None

    data = pd.read_csv(hist_path)
    hist = data.copy()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist.set_index('Date', inplace=True)
    hist.index = TimeZone_convert(hist.index)

    return hist

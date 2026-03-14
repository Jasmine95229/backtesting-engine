import pandas as pd


# --- Trading hours by instrument type (all times US/Eastern) ---
#
# type_1: Sunday 17:00 - Friday 16:55  (crypto, FX majors)
# type_2: Sunday 18:00 - Friday 16:45  (indices, commodities) — hour 17 is maintenance
# type_3: Sunday 20:00 - Friday 16:00  (US stocks via .ext)

TRADING_HOURS = {
    'type_1': [
        'BTCUSD', 'ETHUSD', 'LTCUSD', 'DOGEUSD', 'XLMUSD', 'SOLUSD',
        'EOSUSD', 'AVAXUSD', 'POLUSD', 'LINKUSD', 'XTZUSD', 'KSMUSD', 'GBPUSD'
    ],
    'type_2': [
        'NAS100', 'SPX500', 'XAUUSD', 'XAGUSD', 'VOLX', 'US30', 'GER30', 'USOilSpot'
    ],
}


def ticker_tradable(ticker: str) -> str:
    """Classify a ticker into its trading-hours type."""
    if ticker in TRADING_HOURS['type_1']:
        return 'type_1'
    elif ticker in TRADING_HOURS['type_2']:
        return 'type_2'
    elif ticker.split('.')[-1] == 'ext':
        return 'type_3'
    else:
        return 'type_2'


def _parse_timeframe(timeframe: str):
    """Parse a timeframe string (e.g. 'M5', 'H1', '1D') into pandas freq string."""
    tf = timeframe.upper()

    if tf[0].isdigit():
        number, unit = '', ''
        for c in tf:
            if c.isdigit():
                number += c
            else:
                unit += c
        if unit not in ['D', 'W', 'M', 'Y']:
            raise ValueError(f"Invalid unit '{unit}'. Allowed: D, W, M, Y.")
        return f"{number}{unit}", tf
    else:
        unit = tf[0]
        number = tf[1:]
        if unit not in ['H', 'M', 'S']:
            raise ValueError(f"Invalid unit '{unit}'. Allowed: H, M, S.")
        freq_map = {'H': f"{number}H", 'M': f"{number}T", 'S': f"{number}S"}
        return freq_map[unit], tf


def generate_time_list(start_time: str, end_time: str, timeframe: str, ticker) -> pd.DataFrame:
    """Generate a DataFrame of tradable timestamps for a given ticker and timeframe.

    Columns added:
        - Day, Weekday, Hour, Minute: calendar fields
        - tradable: 1=tradable, 0=rest/last candle (int)
        - market_open: True on the first bar of each week

    Tradable labels (set downstream):
        =1  tradable
        =0  rest period / last candlestick
        =2  forward-filled (set by align_hist_price)
        =3  market-open gap (set by recheck_open_tradable)
        =4  statistics lookback period (set by stats_hist_prices)
    """
    pandas_freq, tf = _parse_timeframe(timeframe)

    time_range = pd.date_range(start=start_time, end=end_time, freq=pandas_freq)
    df = pd.DataFrame(index=time_range)
    df = df[df.index < end_time]
    df['Day'] = df.index.day_name()
    df['Weekday'] = df.index.weekday
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute

    tradable_type = ticker_tradable(ticker)

    if tradable_type == 'type_1':
        fx_mask, last_candle, market_open_hour = _type_1_masks(df, tf)
    elif tradable_type == 'type_2':
        fx_mask, last_candle, mask_not, market_open_hour = _type_2_masks(df, tf)
    elif tradable_type == 'type_3':
        fx_mask, last_candle, market_open_hour = _type_3_masks(df, tf)

    df = df[fx_mask]

    # Set tradable flag
    if tradable_type == 'type_2':
        if last_candle is not None:
            df['tradable'] = (~mask_not) & (~last_candle)
        else:
            df['tradable'] = ~mask_not
    else:
        if last_candle is not None:
            df['tradable'] = ~last_candle
        else:
            df['tradable'] = True

    df['tradable'] = df['tradable'].astype(int)
    df['market_open'] = (df['Weekday'] == 6) & (df['Hour'] == market_open_hour)

    return df


def _type_1_masks(df, tf):
    """Crypto/FX: Sunday 17:00 - Friday 16:55 ET."""
    if tf[1] == 'D':
        fx_mask = (df['Weekday'] >= 0) & (df['Weekday'] <= 4)
        last_candle = None
    elif tf[0] == 'H':
        fx_mask = (
            ((df['Weekday'] == 6) & (df['Hour'] >= 17)) |
            ((df['Weekday'] >= 0) & (df['Weekday'] <= 3)) |
            ((df['Weekday'] == 4) & (df['Hour'] < 17))
        )
        last_candle = None
    elif tf[0] == 'M':
        fx_mask = (
            ((df['Weekday'] == 6) & (df['Hour'] >= 17)) |
            ((df['Weekday'] >= 0) & (df['Weekday'] <= 3)) |
            ((df['Weekday'] == 4) & (df['Hour'] <= 15)) |
            ((df['Weekday'] == 4) & (df['Hour'] == 16) & (df['Minute'] <= 55))
        )
        last_candle = (df['Weekday'] == 4) & (df['Hour'] == 16) & (df['Minute'] == 55)

    return fx_mask, last_candle, 17


def _type_2_masks(df, tf):
    """Indices/commodities: Sunday 18:00 - Friday 16:45 ET, hour 17 maintenance."""
    if tf[1] == 'D':
        fx_mask = (df['Weekday'] >= 0) & (df['Weekday'] <= 4)
        last_candle = None
    elif tf[0] == 'H':
        fx_mask = (
            ((df['Weekday'] == 6) & (df['Hour'] >= 18)) |
            ((df['Weekday'] >= 0) & (df['Weekday'] <= 3)) |
            ((df['Weekday'] == 4) & (df['Hour'] < 17))
        )
        last_candle = None
    elif tf[0] == 'M':
        fx_mask = (
            ((df['Weekday'] == 6) & (df['Hour'] >= 18)) |
            ((df['Weekday'] >= 0) & (df['Weekday'] <= 3)) |
            ((df['Weekday'] == 4) & (df['Hour'] <= 15)) |
            ((df['Weekday'] == 4) & (df['Hour'] == 16) & (df['Minute'] <= 45))
        )
        last_candle = (df['Weekday'] == 4) & (df['Hour'] == 16) & (df['Minute'] == 45)

    mask_not = df['Hour'] == 17
    return fx_mask, last_candle, mask_not, 18


def _type_3_masks(df, tf):
    """US stocks (.ext): Sunday 20:00 - Friday 16:00 ET."""
    if tf[1] == 'D':
        fx_mask = (df['Weekday'] >= 0) & (df['Weekday'] <= 4)
        last_candle = None
    elif tf[0] == 'H':
        fx_mask = (
            ((df['Weekday'] == 6) & (df['Hour'] >= 20)) |
            ((df['Weekday'] >= 0) & (df['Weekday'] <= 3)) |
            ((df['Weekday'] == 4) & (df['Hour'] < 17))
        )
        last_candle = (df['Weekday'] == 4) & (df['Hour'] == 16)
    elif tf[0] == 'M':
        fx_mask = (
            ((df['Weekday'] == 6) & (df['Hour'] >= 20)) |
            ((df['Weekday'] >= 0) & (df['Weekday'] <= 3)) |
            ((df['Weekday'] == 4) & (df['Hour'] <= 15)) |
            ((df['Weekday'] == 4) & (df['Hour'] == 16) & (df['Minute'] == 0))
        )
        last_candle = (df['Weekday'] == 4) & (df['Hour'] == 16) & (df['Minute'] == 0)

    return fx_mask, last_candle, 20

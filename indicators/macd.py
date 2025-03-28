def calculate_macd(series, short_period=12, long_period=26, signal_period=9):
    """Tính MACD và đường tín hiệu."""
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

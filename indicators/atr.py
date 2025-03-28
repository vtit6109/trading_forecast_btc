import pandas as pd

def calculate_atr(data, period=14):
    """Tính Average True Range (ATR)."""
    high_low = data["high"] - data["low"]
    high_close = abs(data["high"] - data["close"].shift())
    low_close = abs(data["low"] - data["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

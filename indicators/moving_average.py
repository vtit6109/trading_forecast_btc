import pandas as pd

def calculate_moving_average(series, period):
    """Tính Moving Average (MA)."""
    return series.rolling(window=period).mean()

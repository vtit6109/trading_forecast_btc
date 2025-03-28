def calculate_bollinger_bands(series, period=20, num_std_dev=2):
    """Tính Bollinger Bands."""
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = ma + (num_std_dev * std)
    lower_band = ma - (num_std_dev * std)
    return upper_band, lower_band

def calculate_stochastic_oscillator(data, period=14):
    """Tính Stochastic Oscillator."""
    low_min = data["low"].rolling(window=period).min()
    high_max = data["high"].rolling(window=period).max()
    stochastic = 100 * ((data["close"] - low_min) / (high_max - low_min))
    return stochastic

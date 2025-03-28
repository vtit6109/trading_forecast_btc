def calculate_pivot_points(data):
    """Tính Pivot Points và các mức kháng cự/hỗ trợ."""
    latest_data = data.iloc[-1]
    high, low, close = latest_data["high"], latest_data["low"], latest_data["close"]
    pivot = (high + low + close) / 3
    resistance1 = 2 * pivot - low
    support1 = 2 * pivot - high
    resistance2 = pivot + (high - low)
    support2 = pivot - (high - low)
    return pivot, resistance1, support1, resistance2, support2

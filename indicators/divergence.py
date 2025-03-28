#divergence.py
# File này chứa các hàm liên quan đến việc tính phân kỳ dựa trên các chỉ báo kỹ thuật như RSI và MACD.

import pandas as pd

from .rsi import calculate_rsi
from .macd import calculate_macd


def detect_divergence(data):
    """
    Phát hiện phân kỳ giữa giá và chỉ báo kỹ thuật.

    Args:
        data (pd.DataFrame): Dữ liệu thị trường với các cột 'close', 'RSI', và 'MACD'.

    Returns:
        list: Danh sách các loại phân kỳ ('Bullish Divergence', 'Bearish Divergence', hoặc None).
    """
    divergences = [None]  # Thêm giá trị None cho hàng đầu tiên
    try:
        for i in range(1, len(data)):
            # Giá đóng cửa (close)
            prev_price = data['close'].iloc[i - 1]
            curr_price = data['close'].iloc[i]

            # RSI và MACD hiện tại và trước đó
            prev_rsi = data['RSI'].iloc[i - 1]
            curr_rsi = data['RSI'].iloc[i]
            prev_macd = data['MACD'].iloc[i - 1]
            curr_macd = data['MACD'].iloc[i]

            # Phát hiện phân kỳ
            if curr_price > prev_price and curr_rsi < prev_rsi:
                divergences.append('Bearish Divergence')  # Giá tăng, RSI giảm
            elif curr_price < prev_price and curr_rsi > prev_rsi:
                divergences.append('Bullish Divergence')  # Giá giảm, RSI tăng
            else:
                divergences.append(None)
    except Exception as e:
        print("Lỗi khi phát hiện phân kỳ:", e)
    
    return divergences

def add_divergence_signals(data):
    """
    Thêm tín hiệu phân kỳ vào dữ liệu thị trường.

    Args:
        data (pd.DataFrame): Dữ liệu thị trường với các cột 'close'.

    Returns:
        pd.DataFrame: Dữ liệu đã thêm tín hiệu phân kỳ.
    """
    try:
        # Tính RSI và MACD nếu chưa có
        if 'RSI' not in data.columns:
            data['RSI'] = calculate_rsi(data['close'], 14)
        if 'MACD' not in data.columns:
            data['MACD'], data['Signal'] = calculate_macd(data['close'])

        # Phát hiện phân kỳ
        data['Divergence'] = detect_divergence(data)
        return data
    except Exception as e:
        print("Lỗi khi thêm tín hiệu phân kỳ:", e)
        return pd.DataFrame()

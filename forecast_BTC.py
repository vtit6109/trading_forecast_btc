# ==== Phân tích và dự đoán xu hướng thị trường BTC/USDT ====
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier  # Thêm XGBoost

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.pivot_points import calculate_pivot_points
from indicators.divergence import detect_divergence, add_divergence_signals  # Import từ divergence.py

# Tải biến môi trường từ file .env
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
BINANCE_API_URL = os.getenv("BINANCE_API_URL")
FEAR_GREED_API_URL = os.getenv("FEAR_GREED_API_URL")
COINGECKO_API_URL = os.getenv("COINGECKO_API_URL")
NEWS_API_URL = os.getenv("NEWS_API_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# ==== Hàm tải dữ liệu từ API ====
def fetch_data(pair):
    """Lấy dữ liệu lịch sử từ API."""
    try:
        url = f"{BINANCE_API_URL}?symbol={pair.replace('/', '')}&interval=1h"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", 
                "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df
        else:
            print(f"Lỗi khi tải dữ liệu từ Binance (HTTP {response.status_code}): {response.text}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return pd.DataFrame()

# ==== Lấy thông tin Fear and Greed ====
def fetch_fear_greed():
    """Lấy chỉ số Fear and Greed từ API."""
    try:
        response = requests.get(FEAR_GREED_API_URL)
        if response.status_code == 200:
            data = response.json()["data"][0]
            return data["value"], data["value_classification"]
        else:
            print(f"Lỗi khi lấy Fear and Greed (HTTP {response.status_code}): {response.text}")
            return None, None
    except Exception as e:
        print(f"Lỗi khi lấy Fear and Greed: {e}")
        return None, None

# ==== Lấy thông tin thị trường ====
def fetch_market_data():
    """Lấy vốn hóa thị trường và khối lượng giao dịch từ API CoinGecko."""
    try:
        response = requests.get(COINGECKO_API_URL)
        if response.status_code == 200:
            data = response.json()["data"]
            market_cap = data["total_market_cap"]["usd"]
            volume = data["total_volume"]["usd"]
            return market_cap, volume
        else:
            print(f"Lỗi khi lấy thông tin thị trường (HTTP {response.status_code}): {response.text}")
            return None, None
    except Exception as e:
        print(f"Lỗi khi lấy thông tin thị trường: {e}")
        return None, None
    
# ==== Lấy tin tức liên quan đến Bitcoin ====
def fetch_btc_news():
    """Lấy tin tức mới nhất liên quan đến Bitcoin từ NewsAPI."""
    try:
        url = f"{NEWS_API_URL}?q=bitcoin&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            news = [{"title": article["title"], "url": article["url"]} for article in articles[:5]]  # Lấy 5 tin tức đầu tiên
            return news
        else:
            print(f"Lỗi khi lấy tin tức Bitcoin (HTTP {response.status_code}): {response.text}")
            return []
    except Exception as e:
        print(f"Lỗi khi lấy tin tức Bitcoin: {e}")
        return []
    
# ==== Chuẩn bị dữ liệu ====
def prepare_data(data):
    """Thêm các chỉ báo kỹ thuật RSI, MACD và tín hiệu phân kỳ vào dữ liệu."""
    try:
        data["RSI"] = calculate_rsi(data["close"], 14)
        data["MACD"], data["Signal"] = calculate_macd(data["close"])
        data = add_divergence_signals(data)  # Thêm tín hiệu phân kỳ
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print("Lỗi khi chuẩn bị dữ liệu:", e)
        return pd.DataFrame()


# ==== Phân loại xu hướng ====
def classify_trend(rsi, macd, signal):
    """Phân loại xu hướng thành Tăng, Giảm, hoặc Tích lũy."""
    if abs(macd - signal) < 0.5 and 40 <= rsi <= 60:
        return "TÍCH LŨY ⚖️"
    elif macd > signal and rsi > 60:
        return "TĂNG 📈"
    elif macd < signal and rsi < 40:
        return "GIẢM 📉"
    else:
        return "TÍCH LŨY ⚖️"

# ==== Gửi thông báo lên Discord ====
def send_analysis_to_discord(trend, price, rsi, macd, pivot_points, fear_greed, market_cap, volume, news, divergence):
    """Gửi phân tích đầy đủ lên Discord."""
    try:
        message = (
            f"📈 **Phân tích BTC/USDT (Khung thời gian: H1)** 📉\n\n"
            f"**Dự Báo Xu hướng sắp tới:** {trend}\n"
            f"- Giá hiện tại: {price} USDT\n\n"
            f"**Chỉ báo kỹ thuật:**\n"
            f"- RSI: {rsi:.2f}\n"
            f"- MACD: {macd:.2f}\n"
            f"- Pivot: {pivot_points[0]:.2f}\n"
            f"- Resistance 1: {pivot_points[1]:.2f}\n"
            f"- Support 1: {pivot_points[2]:.2f}\n"
            f"- Resistance 2: {pivot_points[3]:.2f}\n"
            f"- Support 2: {pivot_points[4]:.2f}\n\n"
            f"**Chỉ số thị trường:**\n"
            f"- Fear and Greed: {fear_greed[0]} ({fear_greed[1]})\n"
            f"- Vốn hóa thị trường: {market_cap / 1e9:.2f} tỷ USD\n"
            f"- Khối lượng giao dịch 24h: {volume / 1e9:.2f} tỷ USD\n\n"
            f"**Tín hiệu phân kỳ:**\n"
            f"- {divergence}\n\n"
            f"**Tin tức quan trọng:**\n"
        )
        for item in news:
            message += f"- [{item['title']}]({item['url']})\n"

        # Thêm cảnh báo
        message += (
            "\n\n---\n"
            "⚠️ **BOT thử nghiệm bởi @VuThangIT** ⚠️\n"
            "BOT này được thiết kế để tự động phân tích dữ liệu thị trường BTC/USDT, "
            "Đưa ra xu hướng dự đoán dựa trên các chỉ báo kỹ thuật như RSI, MACD, và Pivot Points. "
            "Ngoài ra, BOT còn cung cấp thông tin chỉ số Fear & Greed, vốn hóa thị trường, khối lượng giao dịch, "
            "và các tin tức quan trọng mới nhất.\n"
            "Hãy lưu ý, đây chỉ là công cụ hỗ trợ tham khảo. "
            "Cân nhắc kỹ lưỡng trước khi đầu tư và tìm đến chuyên gia để được tư vấn chi tiết hơn."
        )

        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
        if response.status_code == 204:
            print("Đã gửi phân tích lên Discord.")
        else:
            print("Lỗi khi gửi phân tích:", response.status_code, response.text)
    except Exception as e:
        print("Lỗi khi gửi phân tích:", e)

# ==== Chạy dự đoán và thông báo ====
def predict_and_notify(model, pair):
    """Dự đoán xu hướng và gửi thông báo."""
    data = fetch_data(pair)
    if not data.empty:
        data = prepare_data(data)
        if not data.empty:
            latest_data = data.iloc[-1][["RSI", "MACD", "Signal", "Divergence"]].values
            rsi = data["RSI"].iloc[-1]
            macd = data["MACD"].iloc[-1]
            signal = data["Signal"].iloc[-1]
            divergence = data["Divergence"].iloc[-1] or "Không có phân kỳ"
            trend = classify_trend(rsi, macd, signal)
            price = data["close"].iloc[-1]
            pivot_points = calculate_pivot_points(data)
            fear_greed = fetch_fear_greed()
            market_cap, volume = fetch_market_data()
            news = fetch_btc_news()
            send_analysis_to_discord(trend, price, rsi, macd, pivot_points, fear_greed, market_cap, volume, news, divergence)
            
# ==== Main loop ====
if __name__ == "__main__":
    historical_data = fetch_data("BTC/USDT")
    processed_data = prepare_data(historical_data)

    # Huấn luyện mô hình với XGBoost
    X = processed_data[["RSI", "MACD", "Signal"]]
    y = (processed_data["close"].shift(-1) > processed_data["close"]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

    while True:
        predict_and_notify(model, "BTC/USDT")
        time.sleep(3600)  # Gửi mỗi 1 tiếng
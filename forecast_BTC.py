# ==== Phân tích và dự đoán xu hướng thị trường BTC/USDT ====
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.pivot_points import calculate_pivot_points
from indicators.divergence import add_divergence_signals

# Tải biến môi trường từ file .env
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
BINANCE_API_URL = os.getenv("BINANCE_API_URL")
FEAR_GREED_API_URL = os.getenv("FEAR_GREED_API_URL")
COINGECKO_API_URL = os.getenv("COINGECKO_API_URL")
NEWS_API_URL = os.getenv("NEWS_API_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# ==== Hàm tải dữ liệu từ API ====
def fetch_data(pair, interval="1h"):
    """Lấy dữ liệu lịch sử từ API với khung thời gian tùy chỉnh."""
    try:
        url = f"{BINANCE_API_URL}?symbol={pair.replace('/', '')}&interval={interval}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time", 
                "quote_asset_volume", "number_of_trades", 
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
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


# ==== Phân tích nhiều khung thời gian ====
def analyze_multiple_timeframes(pair, intervals):
    """Phân tích nhiều khung thời gian."""
    analysis_results = {}
    for interval in intervals:
        data = fetch_data(pair, interval)
        if not data.empty:
            prepared_data = prepare_data(data)
            if not prepared_data.empty:
                latest_data = prepared_data.iloc[-1]
                analysis_results[interval] = {
                    "RSI": latest_data["RSI"],
                    "MACD": latest_data["MACD"],
                    "Signal": latest_data["Signal"],
                    "Price": latest_data["close"],
                    "PivotPoints": calculate_pivot_points(prepared_data),
                }
    return analysis_results

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
    
# ==== Huấn luyện mô hình Machine Learning ====
def train_ml_model(data):
    """Huấn luyện mô hình XGBClassifier dựa trên dữ liệu lịch sử."""
    try:
        # Chuẩn bị dữ liệu
        data["Label"] = (data["close"].shift(-1) > data["close"]).astype(int)  # Gán nhãn: 1 nếu giá tăng, 0 nếu giá giảm
        features = ["RSI", "MACD", "Signal", "high", "low", "close", "volume"]
        X = data[features].iloc[:-1]  # Loại bỏ hàng cuối vì không có nhãn
        y = data["Label"].iloc[:-1]

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện mô hình
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        # Đánh giá mô hình
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Độ chính xác của mô hình: {accuracy:.2f}")

        return model
    except Exception as e:
        print("Lỗi khi huấn luyện mô hình Machine Learning:", e)
        return None

# ==== Dự đoán xu hướng bằng Machine Learning ====
def predict_trend_with_ml(model, data):
    """Dự đoán xu hướng thị trường dựa trên mô hình Machine Learning."""
    try:
        features = ["RSI", "MACD", "Signal", "high", "low", "close", "volume"]
        latest_data = data[features].iloc[-1:]  # Lấy hàng dữ liệu mới nhất
        prediction = model.predict(latest_data)[0]
        return "TĂNG 📈" if prediction == 1 else "GIẢM 📉"
    except Exception as e:
        print("Lỗi khi dự đoán xu hướng bằng Machine Learning:", e)
        return "Không xác định"


# ==== Rút gọn nội dung trong generate_forecast_with_ai ====
def generate_forecast_with_ai(analysis_results, fear_greed, market_cap, volume, news, model=None):
    """Tạo lời dự báo hấp dẫn bằng AI."""
    try:
        forecast = "📊 **Dự báo xu hướng BTC/USDT** 📊\n\n"
        for interval, result in analysis_results.items():
            trend = classify_trend(result["RSI"], result["MACD"], result["Signal"])
            pivot_points = result["PivotPoints"]
            # Định dạng Pivot Points
            pivot_formatted = (
                f"Pivot: {pivot_points[0]:.2f}, \n\n"
                f"Kháng cự 1: {pivot_points[1]:.2f}, \n"
                f"Hỗ Trợ 1: {pivot_points[2]:.2f}, \n"
                f"Kháng cự 2: {pivot_points[3]:.2f}, \n"
                f"Hỗ trợ 2: {pivot_points[4]:.2f} \n"
            )
            # Dự đoán xu hướng bằng Machine Learning (nếu có mô hình)
            trend_ml = predict_trend_with_ml(model, prepare_data(fetch_data("BTC/USDT", interval))) if model else "Không xác định"

            forecast += (
                f"- **Khung {interval.upper()}**:\n"
                f"  - Xu hướng (Chỉ báo): {trend}\n"
                f"  - Xu hướng (ML): {trend_ml}\n"
                f"  - Giá hiện tại: {result['Price']:.2f} USDT\n"
                f"  - RSI: {result['RSI']:.2f}\n"
                f"  - MACD: {result['MACD']:.2f}\n"
                f"  - Pivot Points: {pivot_formatted}\n\n"
            )
        forecast += (
            f"🌍 **Thông tin thị trường:**\n"
            f"- Chỉ số Fear and Greed: {fear_greed[0]} ({fear_greed[1]})\n"
            f"- Vốn hóa thị trường: {market_cap / 1e9:.2f} tỷ USD\n"
            f"- KL dịch 24h: {volume / 1e9:.2f} tỷ USD\n\n"
            f"📰 **Tin tức nổi bật:**\n"
        )
        # Rút gọn tin tức chỉ lấy 3 tin đầu tiên
        for item in news[:3]:
            forecast += f"- [{item['title']}]({item['url']})\n"

        forecast += (
            "\n\n---\n"
            "⚠️ **BOT thử nghiệm bởi @VuThangIT** ⚠️\n"
            "BOT này được thiết kế để tự động phân tích dữ liệu thị trường BTC/USDT, \n"
            "Đưa ra xu hướng dự đoán dựa trên các chỉ báo kỹ thuật Và Mô hình học máy.\n"
            "Cung cấp thông tin chỉ số Fear & Greed, vốn hóa thị trường, khối lượng giao dịch, "
            "và các tin tức quan trọng mới nhất.\n"
            "⚠️Lưu ý, đây chỉ là công cụ hỗ trợ tham khảo."
            "Cân nhắc kỹ lưỡng trước khi đầu tư và tìm đến chuyên gia để được tư vấn chi tiết hơn."
        )
        return forecast
    except Exception as e:
        print("Lỗi khi tạo lời dự báo bằng AI:", e)
        return "Không thể tạo lời dự báo."

# ==== Gửi thông báo lên Discord ====
def send_forecast_to_discord(forecast):
    """Gửi lời dự báo lên Discord, chia nhỏ nếu vượt quá giới hạn ký tự."""
    try:
        # Discord giới hạn nội dung ở 2000 ký tự
        max_length = 2000
        if len(forecast) > max_length:
            # Chia nhỏ thông báo thành các phần
            parts = [forecast[i:i + max_length] for i in range(0, len(forecast), max_length)]
            for part in parts:
                response = requests.post(DISCORD_WEBHOOK_URL, json={"content": part})
                if response.status_code == 204:
                    print("Đã gửi một phần dự báo lên Discord.")
                else:
                    print("Lỗi khi gửi dự báo:", response.status_code, response.text)
        else:
            # Gửi thông báo nếu không vượt quá giới hạn
            response = requests.post(DISCORD_WEBHOOK_URL, json={"content": forecast})
            if response.status_code == 204:
                print("Đã gửi dự báo lên Discord thành công.")
            else:
                print("Lỗi khi gửi dự báo:", response.status_code, response.text)
    except Exception as e:
        print("Lỗi khi gửi dự báo:", e)

# ==== Main loop ====
if __name__ == "__main__":
    intervals = ["1h", "4h", "1d"]  # Các khung thời gian cần phân tích
    model = None  # Khởi tạo mô hình Machine Learning
    while True:
        analysis_results = analyze_multiple_timeframes("BTC/USDT", intervals)
        if analysis_results:
            fear_greed = fetch_fear_greed()
            market_cap, volume = fetch_market_data()
            news = fetch_btc_news()

            # Huấn luyện mô hình nếu chưa có
            if model is None:
                data = fetch_data("BTC/USDT", "1h")  # Lấy dữ liệu khung 1h để huấn luyện
                prepared_data = prepare_data(data)
                if not prepared_data.empty:
                    model = train_ml_model(prepared_data)

            # Tạo lời dự báo
            forecast = generate_forecast_with_ai(analysis_results, fear_greed, market_cap, volume, news, model)
            send_forecast_to_discord(forecast)
        time.sleep(3600)  # Gửi mỗi 1 tiếng
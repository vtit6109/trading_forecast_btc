# Trading Bot for Forecasting BTC

## Description
This project is a trading bot designed to forecast Bitcoin (BTC) prices using technical indicators such as RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence). The bot is built in Python and supports modular functionality for calculating indicators, detecting divergences, and adding trading signals.

---

## Features
- **RSI Calculation**: Calculates the RSI of the given price data.
- **MACD Calculation**: Computes the MACD and its Signal line.
- **Divergence Detection**: Identifies bullish and bearish divergences.
- **Modular Structure**: Includes reusable functions for indicators and signal detection.
- **Custom Configuration**: Use your own data and customize indicator parameters.

---

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vtit6109/trading_forecast_btc.git
   cd trading_bot
   ```

2. **Install Dependencies**:
   Ensure Python 3 is installed. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. **Prepare Data**: Ensure your data includes a column named `close` for closing prices.
2. **Run the Bot**:
   Execute the bot to calculate indicators and generate signals:
   ```bash
   python forecast_BTC.py
   ```

---

## Project Structure
```
trading_bot/
├── forecast_BTC.py          # Main script to forecast BTC prices
├── indicators/
│   ├── rsi.py              # RSI calculation module
│   ├── macd.py             # MACD calculation module
│   └── divergence.py       # Divergence detection functions
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

---

## Configuration
Edit `forecast_BTC.py` to customize settings such as indicator parameters or data file paths.

---

## Contact
If you have any questions or suggestions, please contact:
- **Email**: [your_email@example.com]
- **GitHub**: [https://github.com/vtit6109](https://github.com/vtit6109)


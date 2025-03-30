# 📈 Stock Price Predictor

A robust machine learning application designed to predict stock prices using **LSTM (Long Short-Term Memory)** neural networks. Built using **Streamlit**, **TensorFlow**, and **yfinance** for a seamless user experience.

---

## 🚀 Features

- **Real-Time Data Fetching**: Automatically fetch historical stock data with `yfinance`.
- **Interactive Interface**: User-friendly and customizable Streamlit interface.
- **LSTM-Based Predictions**: Utilize deep learning for accurate stock price forecasting.
- **Customizable Parameters**: Tailor prediction settings to fit your analysis needs.
- **Visual Insights**: Dynamic charts for historical and predicted price trends.

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## ▶️ Usage

1. Run the Streamlit app:
```bash
streamlit run stock_predictor.py
```

2. Enter a stock symbol (e.g., AAPL, GOOGL)
3. Select the date range and prediction parameters
4. Click "Predict" to see the results

## ⚙️ Parameters

- **Stock Ticker**: Symbol of the stock to predict
- **Start Date**: Historical data start date
- **Days to Use for Prediction**: Number of past days to use for training
- **Days to Predict**: Number of future days to predict

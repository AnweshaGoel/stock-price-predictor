# Stock Price Predictor

A machine learning application that predicts stock prices using LSTM (Long Short-Term Memory) neural networks. Built with Streamlit, TensorFlow, and yfinance.

## Features

- Real-time stock data fetching using yfinance
- Interactive UI with Streamlit
- LSTM-based price prediction
- Customizable prediction parameters
- Visual representation of historical prices and predictions
- Model performance metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run stock_predictor.py
```

2. Enter a stock symbol (e.g., AAPL, GOOGL)
3. Select the date range and prediction parameters
4. Click "Predict" to see the results

## Parameters

- **Stock Ticker**: Symbol of the stock to predict
- **Start Date**: Historical data start date
- **Days to Use for Prediction**: Number of past days to use for training
- **Days to Predict**: Number of future days to predict

## Screenshots

[Add screenshots of your application here]

## Disclaimer

This application is for educational purposes only. Stock market predictions are inherently uncertain and should not be the sole basis for investment decisions.

## License

MIT License

## Author

[Your Name] 
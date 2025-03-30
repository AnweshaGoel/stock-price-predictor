import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.title("Stock Price Predictor")
st.markdown("""
This application predicts future stock prices using LSTM (Long Short-Term Memory) neural networks.
Enter a stock ticker symbol and prediction parameters below.
""")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
prediction_days = st.sidebar.slider("Days to Use for Prediction", 30, 300, 60)
future_days = st.sidebar.slider("Days to Predict into Future", 1, 30, 7)

@st.cache_data
def load_data(ticker, start_date):
    end_date = datetime.datetime.now().date()
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def prepare_data(data, prediction_days):
    # Check if data exists
    if data is None or len(data) < prediction_days:
        st.error("Insufficient data for the selected parameters")
        return None, None, None, None
    
    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    # Create training dataset
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
    
    # Convert lists to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Reshape data for LSTM input (samples, time steps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, scaled_data

def build_model(x_train, y_train):
    # Check if training data exists
    if x_train is None or y_train is None:
        return None, None
    
    # Clear any previous Keras/TF session to prevent name scope errors
    tf.keras.backend.clear_session()
    
    # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model with progress bar
    with st.spinner('Training model... This may take a few minutes.'):
        history = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0)
    
    return model, history

def predict_future(model, data, scaler, prediction_days, future_days):
    # Check if model or data is None
    if model is None or data is None:
        return None
    
    # Get the last 'prediction_days' of scaled data
    inputs = data['Close'].values[-prediction_days:].reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    # Initialize list to store predictions
    future_predictions = []
    current_batch = inputs.reshape((1, prediction_days, 1))
    
    # Make predictions for 'future_days'
    for _ in range(future_days):
        # Get prediction for next day
        current_pred = model.predict(current_batch, verbose=0)[0]
        future_predictions.append(current_pred[0])
        
        # Update the batch for next prediction (remove first element and add the prediction)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    
    # Inverse transform the predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions

def main():
    # Load data
    data = load_data(ticker, start_date)
    if data is None or data.empty:
        st.error("No data available. Please check your ticker symbol and date range.")
        return
    
    # Display basic info
    st.subheader(f"Data for {ticker}")
    st.write(data.tail())
    
    # Plot the historical stock price
    st.subheader("Historical Stock Price")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'])
    ax.set_title(f'{ticker} Close Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)
    
    # Prepare data
    x_train, y_train, scaler, scaled_data = prepare_data(data, prediction_days)
    
    # Check if data preparation was successful
    if x_train is None:
        return
    
    # Build and train model
    model, history = build_model(x_train, y_train)
    
    # Check if model building was successful
    if model is None:
        return
    
    # Plot training loss
    st.subheader("Model Training Loss")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.history['loss'])
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    st.pyplot(fig)
    
    # Make predictions on test data
    with st.spinner('Making predictions...'):
        # Prepare data for testing (last prediction_days)
        test_data = data.copy()
        actual_prices = test_data['Close'].values
        
        # Make predictions for future days
        future_predictions = predict_future(model, test_data, scaler, prediction_days, future_days)
        
        # Check if predictions were successful
        if future_predictions is None:
            st.error("Failed to generate predictions.")
            return
        
        # Create dates for future predictions
        last_date = test_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        # Create a DataFrame for predictions
        prediction_df = pd.DataFrame(data={
            'Predicted Price': future_predictions.flatten()
        }, index=future_dates)
        
        # Display predictions
        st.subheader(f"Price Predictions for Next {future_days} Days")
        st.write(prediction_df)
        
        # Plot predictions
        st.subheader("Historical Prices with Predictions")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical prices
        ax.plot(test_data.index, actual_prices, label='Actual Prices')
        
        # Plot future predictions
        ax.plot(future_dates, future_predictions, 'r--', label='Predicted Prices')
        
        # Add a vertical line to indicate where predictions start
        ax.axvline(x=last_date, color='g', linestyle='--')
        
        ax.set_title(f'{ticker} Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)
        
        # Disclaimer
        st.warning("""
        **Disclaimer**: These predictions are for educational purposes only. 
        Stock market predictions are inherently uncertain and should not be the sole basis for investment decisions.
        """)

if __name__ == "__main__":
    main()
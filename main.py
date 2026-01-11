import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and Clean Data
# The AAPL.csv has specialized headers, we skip the first 3 lines
column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df = pd.read_csv('AAPL.csv', skiprows=3, names=column_names)

# Convert Date to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').dropna()

# 2. Feature Engineering
# Adding simple technical indicators to improve prediction
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df = df.dropna() # Remove rows with NaN from rolling window

# Select features for the model
# Using Close price and SMAs as inputs
features = ['Close', 'SMA_20', 'SMA_50']
data = df[features].values

# 3. Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Create Sequences (Sliding Window)
def create_sequences(data, window_size=60):
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data[i-window_size:i, :]) # Multiple features
        y.append(data[i, 0]) # Target is the next 'Close' price
    return np.array(x), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

# 5. Chronological Train-Test Split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Build Bidirectional LSTM Model
model = Sequential([
    # First BiLSTM layer
    Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    
    # Second BiLSTM layer
    Bidirectional(LSTM(units=50)),
    Dropout(0.2),
    
    # Fully connected layers
    Dense(units=25, activation='relu'),
    Dense(units=1) # Predicting the single Close price
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Training
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 8. Evaluation & Inverse Transformation
predictions = model.predict(X_test)

# To inverse transform, we need to create a dummy array with same shape as 'features'
prediction_copies = np.repeat(predictions, len(features), axis=-1)
y_pred_inv = scaler.inverse_transform(prediction_copies)[:,0]

y_test_copies = np.repeat(y_test.reshape(-1, 1), len(features), axis=-1)
y_test_inv = scaler.inverse_transform(y_test_copies)[:,0]

# 9. Visualization
plt.figure(figsize=(14, 7))
test_dates = df['Date'].iloc[-len(y_test_inv):]
plt.plot(test_dates, y_test_inv, color='blue', label='Actual AAPL Price')
plt.plot(test_dates, y_pred_inv, color='red', linestyle='--', label='BiLSTM Predicted Price')
plt.title('AAPL Stock Price Prediction - Bidirectional LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
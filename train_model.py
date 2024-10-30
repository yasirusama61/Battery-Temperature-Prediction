#!/usr/bin/env python
# coding: utf-8

"""
Battery Temperature Prediction Pipeline
Author: Usama Yasir Khan
Date: 2024-10-30

This script performs the following steps:
1. Data Loading and Preprocessing
2. Feature Engineering
3. Data Resampling and Interpolation
4. Train-Validation-Test Split
5. Data Normalization and Sequence Creation
6. LSTM Model Training and Evaluation
7. Model Evaluation Metrics and Visualization

The primary objective is to predict battery temperature using LSTM with engineered features, rolling statistics, and interaction terms.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px

# 1. Data Loading and Initial Processing
data = pd.read_parquet("extracted_battery_data.parquet")
data['Time Stamp'] = pd.to_datetime(data['Time Stamp'], errors='coerce')
data.set_index('Time Stamp', inplace=True)

# 2. Resampling and Interpolation
def resample_and_interpolate(data, frequency, output_file):
    numeric_data = data.select_dtypes(include=['number'])
    data_resampled = numeric_data.resample(frequency).mean()
    data_resampled = data_resampled.interpolate(method='linear')
    data_resampled.to_parquet(output_file)
    print(f"Data resampled to {frequency} intervals and saved to {output_file}.")

# Resample to 1 Hz and 10s intervals
resample_and_interpolate(data, '1S', "resampled_battery_data_1Hz.parquet")
resample_and_interpolate(data, '10S', "resampled_battery_data_10s.parquet")

# 3. Feature Engineering
data['Temp_Rolling_Mean'] = data['Temperature [C]'].rolling(window=30).mean()
data['Voltage_Rolling_Std'] = data['Voltage [V]'].rolling(window=30).std()
data['Current_Rolling_Mean'] = data['Current [A]'].rolling(window=30).mean()
data['Temp_Lag_1'] = data['Temperature [C]'].shift(1)
data['Voltage_Lag_1'] = data['Voltage [V]'].shift(1)
data['Voltage_Current_Interaction'] = data['Voltage [V]'] * data['Current [A]']
data['Temperature_Current_Interaction'] = data['Temperature [C]'] * data['Current [A]']
data['Cumulative_Capacity'] = data['Capacity [Ah]'].cumsum()
data['Cumulative_WhAccu'] = data['WhAccu [Wh]'].cumsum()
data.dropna(inplace=True)

# 4. Train-Validation-Test Split
train_end = int(0.7 * len(data))
val_end = int(0.85 * len(data))
train_data, val_data, test_data = data[:train_end], data[train_end:val_end], data[val_end:]
print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# 5. Data Normalization and Sequence Creation
features = ['Voltage [V]', 'Current [A]', 'Temperature [C]', 'Temp_Rolling_Mean', 'Voltage_Rolling_Std', 
            'Current_Rolling_Mean', 'Temp_Lag_1', 'Voltage_Lag_1', 
            'Voltage_Current_Interaction', 'Temperature_Current_Interaction',
            'Cumulative_Capacity', 'Cumulative_WhAccu']
target = 'Temperature [C]'

X_train, y_train = train_data[features], train_data[[target]]
X_val, y_val = val_data[features], val_data[[target]]
X_test, y_test = test_data[features], test_data[[target]]

scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
X_train_scaled, X_val_scaled, X_test_scaled = scaler_X.fit_transform(X_train), scaler_X.transform(X_val), scaler_X.transform(X_test)
y_train_scaled, y_val_scaled, y_test_scaled = scaler_y.fit_transform(y_train), scaler_y.transform(y_val), scaler_y.transform(y_test)

window_size = 10
def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, window_size)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, window_size)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)

# 6. LSTM Model Definition and Training
model = Sequential([
    LSTM(64, input_shape=(window_size, len(features))),
    Dense(1)  # Output layer for temperature prediction
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=100, batch_size=32, callbacks=[early_stopping])
model.save("temperature_prediction_lstm.h5")

# 7. Model Evaluation Metrics and Visualization
test_loss, test_mae = model.evaluate(X_test_seq, y_test_seq)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

y_pred = model.predict(X_test_seq)
mse, rmse, r2 = mean_squared_error(y_test_seq, y_pred), np.sqrt(mean_squared_error(y_test_seq, y_pred)), r2_score(y_test_seq, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 8. Plot Actual vs Predicted Temperature
y_test_inv, y_pred_inv = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)), scaler_y.inverse_transform(y_pred)
plt.figure(figsize=(14, 7))
plt.plot(y_test_inv, label='Actual Temperature', color='blue', alpha=0.6)
plt.plot(y_pred_inv, label='Predicted Temperature', color='red', alpha=0.6)
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Samples')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid()
plt.show()

# Save visualizations as HTML files using Plotly for interactive inspection
fig_temp = px.line(data, x=data.index, y='Temperature [C]', title='Temperature Over Time')
fig_temp.update_layout(xaxis_title='Time', yaxis_title='Temperature (°C)')
fig_temp.write_html("temperature_over_time.html")

fig_voltage = px.line(data, x=data.index, y='Voltage [V]', title='Voltage Over Time')
fig_voltage.update_layout(xaxis_title='Time', yaxis_title='Voltage (V)')
fig_voltage.write_html("voltage_over_time.html")

fig_current = px.line(data, x=data.index, y='Current [A]', title='Current Over Time')
fig_current.update_layout(xaxis_title='Time', yaxis_title='Current (A)')
fig_current.write_html("current_over_time.html")

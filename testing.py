# Validation and Testing Script

# Author: Usama Yasir Khan
# Owner: Usama Yasir Khan, AI Engineer specializing in battery management systems and predictive modeling.

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('temperature_model.h5')

# Load the scalers
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# 1. Load New Data
data = pd.read_csv('new_validation_data.csv')

# 2. Resample Data to 10-Second Intervals
data['Time [s]'] = pd.to_timedelta(data['Time [s]'], unit='s')
data.set_index('Time [s]', inplace=True)
data = data.resample('10S').mean()
data.reset_index(inplace=True)
data['Time [s]'] = data['Time [s]'].dt.total_seconds()

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

# 4. Create Sequences for LSTM Model
sequence_length = 10
sequences = []
labels = []

for i in range(len(data) - sequence_length):
    seq = data.iloc[i:i + sequence_length][[
        'Voltage [V]', 'Current [A]', 'Temperature [C]', 'Temp_Rolling_Mean',
        'Voltage_Rolling_Std', 'Current_Rolling_Mean', 'Temp_Lag_1', 'Voltage_Lag_1',
        'Voltage_Current_Interaction', 'Temperature_Current_Interaction',
        'Cumulative_Capacity', 'Cumulative_WhAccu']].values
    label = data.iloc[i + sequence_length]['Temperature [C]']
    sequences.append(seq)
    labels.append(label)

X_test = np.array(sequences)
y_test = np.array(labels).reshape(-1, 1)

# Scale the data
X_test_scaled = scaler_x.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# 5. Model Prediction
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 6. Save Predictions
data = data.iloc[sequence_length:]
data['Predicted Temperature [C]'] = y_pred

data.to_csv('validated_results.csv', index=False)
print("Validation and predictions complete. Results saved to 'validated_results.csv'")

# 7. Plot Actual vs. Predicted Temperature
plt.figure(figsize=(14, 6))
plt.plot(data['Time [s]'], data['Temperature [C]'], label='Actual Temperature', color='blue')
plt.plot(data['Time [s]'], data['Predicted Temperature [C]'], label='Predicted Temperature', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.title('Actual vs. Predicted Temperature')
plt.legend()
plt.show()

# 8. Residual Analysis
residuals = data['Temperature [C]'] - data['Predicted Temperature [C]']

# Plot residuals over time
plt.figure(figsize=(14, 6))
plt.plot(data['Time [s]'], residuals, marker='o', linestyle='', markersize=3, color='red', label='Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residuals Over Time')
plt.legend()
plt.show()

# Plot histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residual (Error)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# Unit Tests
class TestValidationScript(unittest.TestCase):
    def test_feature_engineering(self):
        # Test if feature engineering creates the correct columns
        self.assertIn('Temp_Rolling_Mean', data.columns)
        self.assertIn('Voltage_Rolling_Std', data.columns)
        self.assertIn('Current_Rolling_Mean', data.columns)
        self.assertIn('Temp_Lag_1', data.columns)
        self.assertIn('Voltage_Lag_1', data.columns)
        self.assertIn('Voltage_Current_Interaction', data.columns)
        self.assertIn('Temperature_Current_Interaction', data.columns)
        self.assertIn('Cumulative_Capacity', data.columns)
        self.assertIn('Cumulative_WhAccu', data.columns)

    def test_sequence_creation(self):
        # Test if sequences are created correctly
        self.assertEqual(X_test_seq.shape[1], sequence_length)
        self.assertEqual(X_test_seq.shape[2], len(feature_columns))

    def test_data_scaling(self):
        # Test if data scaling maintains correct shape
        self.assertEqual(X_test_scaled.shape, (X_test_seq.shape[0], sequence_length, len(feature_columns)))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

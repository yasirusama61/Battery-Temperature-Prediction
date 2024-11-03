# Validation and Testing Script

# Author: Usama Yasir Khan
# Owner: Usama Yasir Khan, AI Engineer specializing in battery management systems and predictive modeling.

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

# Load the saved model
model = tf.keras.models.load_model('saved_model.h5')

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

# 4. Prepare Data for Model Testing
feature_columns = [
    'Voltage [V]', 'Current [A]', 'Temperature [C]', 'Temp_Rolling_Mean',
    'Voltage_Rolling_Std', 'Current_Rolling_Mean', 'Temp_Lag_1', 'Voltage_Lag_1',
    'Voltage_Current_Interaction', 'Temperature_Current_Interaction',
    'Cumulative_Capacity', 'Cumulative_WhAccu'
]

X_test = data[feature_columns]
X_test_scaled = scaler_x.transform(X_test)

# 5. Create Sequences for LSTM Input
sequence_length = 10
X_test_seq = []
for i in range(len(X_test_scaled) - sequence_length + 1):
    X_test_seq.append(X_test_scaled[i:i + sequence_length])
X_test_seq = np.array(X_test_seq)

# 6. Model Prediction
y_pred_scaled = model.predict(X_test_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 7. Save Predictions
data = data.iloc[sequence_length - 1:]
data['Predicted Temperature [C]'] = y_pred

data.to_csv('validated_results.csv', index=False)
print("Validation and predictions complete. Results saved to 'validated_results.csv'")

# 8. Calculate and Print Performance Metrics
mse = mean_squared_error(data['Temperature [C]'], y_pred)
r2 = r2_score(data['Temperature [C]'], y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
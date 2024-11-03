# Validation and Model Testing Script

"""
Author: Usama Yasir Khan
AI Engineer specializing in battery management systems and predictive modeling.
This script is developed as part of the temperature prediction project for validating new data and testing the trained model.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model
model = joblib.load('temperature_model.pkl')

# Load new data
new_data = pd.read_csv('new_validation_data.csv')

# Convert 'Time [s]' column to datetime for resampling
new_data['Time [s]'] = pd.to_timedelta(new_data['Time [s]'], unit='s')
new_data.set_index('Time [s]', inplace=True)

# Resample data to 10-second intervals
new_data_resampled = new_data.resample('10S').mean()
new_data_resampled = new_data_resampled.interpolate()  # Handle any NaNs

# Create new features
new_data_resampled['Cumulative_Capacity'] = (new_data_resampled['Current [A]'] * (10 / 3600)).cumsum()  # Capacity in Ah
new_data_resampled['Cumulative_WhAccu'] = (new_data_resampled['Voltage [V]'] * new_data_resampled['Current [A]'] * (10 / 3600)).cumsum()  # Energy in Wh

# Ensure SOC column exists (if needed, calculate it)
if 'SOC' not in new_data_resampled.columns:
    nominal_capacity = 5  # Adjust based on your cell capacity
    new_data_resampled['SOC'] = 1 - (new_data_resampled['Cumulative_Capacity'] / nominal_capacity)

# Prepare data for model input (ensure feature alignment)
features = ['Voltage [V]', 'Current [A]', 'Temperature [C]', 'Cumulative_Capacity', 'Cumulative_WhAccu', 'SOC']
X_new = new_data_resampled[features].values

# Make predictions
predictions = model.predict(X_new)

# Evaluate the model
actuals = new_data_resampled['Temperature [C]'].values  # Replace with actual target column if different
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actuals, predictions)

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Plot actual vs predicted temperatures
plt.figure(figsize=(14, 6))
plt.plot(actuals, label='Actual Temperature', color='blue')
plt.plot(predictions, label='Predicted Temperature', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Temperature [C]')
plt.title('Model Predictions vs Actual Temperature')
plt.legend()
plt.show()

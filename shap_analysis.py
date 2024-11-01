"""
SHAP Analysis Script for Battery Temperature Prediction
Author: Usama Yasir Khan
Designation: AI Engineer at Xing Mobility

Description:
This script performs SHAP (SHapley Additive exPlanations) analysis to interpret the LSTM-based model used for predicting 
battery temperature. The script provides insights into feature importance and interactions, helping to understand 
which features most significantly influence temperature predictions.

Features included in the analysis:
- Voltage [V]
- Current [A]
- Temperature [C]
- Various engineered features such as rolling means, lagged values, interaction terms, and cumulative metrics

The script includes:
1. Loading the pre-trained model.
2. Initializing SHAP's GradientExplainer with sample training data.
3. Calculating SHAP values for a test dataset sample.
4. Generating visualizations to demonstrate global feature importance and specific feature interactions.

Usage:
This script can be run in a Python environment where TensorFlow, SHAP, and matplotlib are installed. 
Make sure that the paths to the model and any data files are correct before running.

Note: This project is owned by Usama Yasir Khan as part of the battery management system (BMS) project at Xing Mobility.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model (make sure the path is correct)
model = load_model("temperature_prediction_lstm.h5")

# Define feature names based on your feature engineering process
feature_names = [
    'Voltage [V]', 'Current [A]', 'Temperature [C]', 
    'Temp_Rolling_Mean', 'Voltage_Rolling_Std', 
    'Current_Rolling_Mean', 'Temp_Lag_1', 'Voltage_Lag_1', 
    'Voltage_Current_Interaction', 'Temperature_Current_Interaction',
    'Cumulative_Capacity', 'Cumulative_WhAccu'
]

# Step 1: Sample Data Selection
# Assuming you have loaded or split `X_train` and `X_test` data elsewhere
# Take a small sample from the training data to initialize SHAP
sample_train_data = X_train_seq[:100]  # (100, 10, 12) for SHAP background
sample_test_data = X_test_seq[:50]     # (50, 10, 12) for SHAP evaluation

# Step 2: Initialize GradientExplainer
explainer = shap.GradientExplainer(model, sample_train_data)

# Step 3: Compute SHAP values for test data
# Note: GradientExplainer requires a 2D or 3D input, so we flatten accordingly
sample_test_data_flattened = sample_test_data.reshape(sample_test_data.shape[0], -1)
shap_values = explainer.shap_values(sample_test_data_flattened)

# Step 4: Visualizing SHAP Results

# 1. Summary Plot - Global Feature Importance
shap_values_flattened = np.array([sv.flatten() for sv in shap_values[0]])  # Flatten for summary plot
plt.figure()
shap.summary_plot(shap_values_flattened, sample_test_data_flattened, feature_names=feature_names)
plt.savefig("summary_plot.png")  # Save the plot if needed

# 2. Dependence Plot - Interaction between Temperature and Voltage
plt.figure()
shap.dependence_plot("Temperature [C]", shap_values_flattened, sample_test_data_flattened, feature_names=feature_names, interaction_index="Voltage [V]")
plt.savefig("dependence_plot_temp_voltage.png")

# 3. Optional Force Plot for Single Prediction
# Choose a single sample and select a single timestep for visualization
baseline_prediction = model.predict(np.expand_dims(sample_test_data[0], axis=0)).mean()
shap.force_plot(
    baseline_prediction,
    shap_values_flattened[0],  # SHAP values for first prediction
    sample_test_data_flattened[0],  # Actual input for first sample
    feature_names=feature_names
)

# Show plots
plt.show()

print("SHAP analysis completed and visualizations saved.")
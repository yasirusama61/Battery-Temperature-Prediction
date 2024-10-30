# Battery Temperature Prediction

This project focuses on predicting the temperature of an LG 18650HG2 battery cell based on various parameters like voltage, current, and cycle information. The goal is to develop a reliable temperature prediction model that aids battery management systems by anticipating overheating risks and enhancing battery performance.

## Project Overview

Battery temperature plays a critical role in the safety and efficiency of lithium-ion cells. This project employs advanced modeling techniques to predict temperature using time-series data, with unique features engineered to capture cycle-specific behaviors, accumulated energy, and thermal changes.

## Data Description

### Original Data Source
The dataset used in this project originates from research conducted at McMaster University, Ontario, Canada, and is publicly available on Mendeley Data ([link](https://data.mendeley.com/datasets/cp3473x7xv/2)). This data was collected by Dr. Phillip Kollmeyer and colleagues as part of their work on State-of-Charge (SOC) estimation for lithium-ion batteries using a deep feedforward neural network (FNN) approach.

**Original Data Citation**:
- Philip Kollmeyer, Carlos Vidal, Mina Naguib, Michael Skells. *LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script*. Published: March 6, 2020. DOI: [10.17632/cp3473x7xv.3](https://doi.org/10.17632/cp3473x7xv.3)

The data was collected using an LG 18650HG2 battery cell, tested in a thermal chamber and analyzed with a Digatron Universal Battery Tester.

### Summary Statistics of Data

Below is a summary of the main variables in the dataset, including key statistical indicators (mean, standard deviation, min, max, and quartiles):

| Statistic      | Step         | Voltage [V] | Current [A] | Temperature [C] | Capacity [Ah] | WhAccu [Wh] |
|----------------|--------------|-------------|-------------|-----------------|---------------|-------------|
| **Count**      | 5,423,272    | 5,423,272   | 5,423,272   | 5,423,272       | 5,423,272     | 5,423,272   |
| **Mean**       | 351.87       | 3.912       | 8.52e-3     | 8.35            | 0.1139        | 0.5481      |
| **Std**        | 799.02       | 0.3629      | 1.0469      | 16.6851         | 1.3026        | 4.9892      |
| **Min**        | 1            | 2.262       | -1.8001     | -22.5039        | -2.7807       | -10.3028    |
| **25%**        | 9            | 3.665       | -0.0407     | -9.6814         | -0.5800       | -2.1959     |
| **50%**        | 24           | 4.151       | 0.0         | 9.1625          | 0.0105        | 0.0457      |
| **75%**        | 44           | 4.187       | 0.0         | 23.8307         | 0.9399        | 3.6855      |
| **Max**        | 6,667        | 4.202       | 5.9993      | 41.3273         | 2.8180        | 10.9525     |

### Observations and Analysis

1. **Voltage**:
   - The voltage has a mean of around 3.91 V, which is typical for an LG 18650 battery under operation.
   - The minimum voltage recorded is 2.26 V, likely representing near-discharged conditions, while the maximum of 4.20 V aligns with a fully charged state.
   - Standard deviation is low, indicating voltage remains relatively stable during most of the cycle.

2. **Current**:
   - The mean current is near zero (0.0085 A), which suggests the dataset contains balanced charge and discharge cycles.
   - A wide range of current values, from -1.80 A (indicating discharge) to 5.99 A (indicating charge), shows various testing conditions. Negative values denote discharge events.
   - The relatively higher standard deviation of 1.05 A signifies that current changes frequently, possibly due to dynamic drive cycles or charge/discharge tests.

3. **Temperature**:
   - Temperature has a broad range from -22.50 °C to 41.32 °C, indicating testing under various ambient conditions.
   - The mean temperature (8.35 °C) and high standard deviation (16.68 °C) suggest temperature fluctuations due to different test conditions, including extreme high and low-temperature tests.
   - The presence of very low temperatures (below freezing) could imply cold weather performance tests.

4. **Capacity (Ah)**:
   - The mean capacity is 0.1139 Ah, which might reflect intermittent data capture across charge and discharge cycles.
   - A negative minimum value for capacity (-2.78 Ah) is unusual, possibly due to data handling or measurement anomalies that will need addressing.
   - The maximum capacity (2.81 Ah) aligns with the expected capacity of a fully charged 3Ah LG 18650 cell, suggesting full discharge cycles were included.

5. **Accumulated Energy (WhAccu)**:
   - The mean accumulated energy is 0.548 Wh, with a range spanning -10.30 Wh to 10.95 Wh.
   - Negative values might correspond to discharge cycles, where energy is drawn from the battery.
   - The range and standard deviation indicate varied usage scenarios, which could enhance the robustness of a predictive model by covering multiple operational conditions.

### Insights for Modeling
- The broad ranges and fluctuations in temperature, current, and capacity are beneficial for developing a robust temperature prediction model as they cover a wide range of realistic battery conditions.
- Handling negative values in capacity and accumulated energy may require preprocessing, as these could be anomalies or represent specific states in the cycle.
- Since the data includes both high and low temperatures, it may be useful to segment or engineer features for extreme conditions to improve model performance.

## Exploratory Data Analysis (EDA)

### Temperature Over Time
The temperature plot shows notable changes and fluctuations over time:
- **Early Stability**: Initially, the temperature is stable, indicating a relatively constant operating condition.
- **Sudden Increases**: There are points where the temperature rises significantly, reaching above 40°C, likely due to charging or heavy usage phases.
- **Gradual Cooling**: After peak activity, there is a gradual decrease in temperature, potentially corresponding to cooling periods or rest phases.
- **Dynamic Cycles**: We observe multiple cycles of heating and cooling, indicative of repeated charging/discharging sessions. These fluctuations can be used to analyze battery behavior under various loads and operational conditions.

![Temperature Over Time](plots/temperature_over_time.png)

### Voltage Over Time
The voltage plot reveals essential patterns related to battery charge cycles:
- **Repetitive Patterns**: There are recurring voltage cycles, possibly corresponding to charging and discharging states.
- **Voltage Drops**: Frequent voltage drops to around 3V indicate discharge events, while rises to over 4V likely indicate charging sessions.
- **Overall Stability**: Despite fluctuations, the voltage remains within the typical operational range for lithium-ion cells, showing consistent behavior across cycles.

![Voltage Over Time](plots/voltage_over_time.png)

### Current Over Time
The current plot offers insights into the battery’s power usage and regeneration:
- **Positive and Negative Peaks**: Positive peaks represent charging currents, while negative peaks indicate discharging currents.
- **High Discharge Currents**: The plot shows high discharge rates during certain intervals, which could correspond to high-power consumption phases.
- **Low to Zero Current Phases**: These are indicative of rest or idle periods between active charge/discharge cycles.

![Current Over Time](plots/current_overtime.png)

This analysis offers a foundational understanding of the battery’s performance over time and can inform the feature engineering and model-building stages.


### Project-Specific Data Processing
The data includes the following fields:
- **Time Stamp**: Timestamp for each data entry.
- **Cycle**, **Step Time**, **Procedure**: Information about charge/discharge cycles.
- **Voltage**, **Current**: Electrical parameters influencing temperature.
- **Temperature**: The target variable for prediction.

## Project Goals
1. **Data Resampling and Consistency**: Resample data at a consistent 1 Hz frequency to handle varying sample rates in the original dataset.
2. **Feature Engineering**: Create cycle-level, cumulative, and interaction features to capture underlying patterns.
3. **Model Development**: Develop and train sequential models (e.g., LSTM) for temperature prediction.
4. **Evaluation**: Assess the model’s accuracy and explore optimization opportunities.
5. **Dashboard Creation**: Develop a visualization tool for real-time temperature monitoring.

## Repository Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `src/`: Scripts for data processing and model training.
- `models/`: Saved models.
- `results/`: Plots, metrics, and other evaluation outputs.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yasirusama61/Battery-Temperature-Prediction.git
   cd Battery-Temperature-Prediction

## Installation and Usage

### Install dependencies:
To install the necessary packages, run:
```bash
   pip install -r requirements.txt
```
### Run the data extraction and processing script:
Execute the script to extract and preprocess the raw battery data:
```bash
   python src/data_processing.py
```
### Train the model (example command):
Train the temperature prediction model on the preprocessed data:
```bash
   python src/train_model.py
```
### Launch the dashboard for real-time monitoring (optional):
If you'd like to visualize real-time temperature predictions and data insights, launch the dashboard:
```bash
   python src/dashboard.py
```

# Model Architecture

The model architecture chosen for this project is a simple but effective Long Short-Term Memory (LSTM) network, specifically designed for sequential data and time-series prediction. The architecture includes:

- **LSTM Layer**: This layer consists of 64 units and serves as the core of the model. It is well-suited for capturing temporal dependencies in time-series data, which is essential for predicting battery temperature based on historical data points.
- **Dense Layer**: A single Dense layer with one unit forms the output layer, responsible for generating the temperature prediction.

The choice of an LSTM layer allows the model to learn patterns in the data that occur over time, making it capable of understanding cycles and other temporal relationships within the battery parameters.

# Model Code

Below is the code used to define and train the model:

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Define the model
model = Sequential([
    LSTM(64, input_shape=(window_size, len(features))),
    Dense(1)  # Output layer for temperature prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the trained model
model.save("temperature_prediction_lstm.h5")
```

# Model Training and Validation

During training, the **Mean Squared Error (MSE)** was used as the loss function, while **Mean Absolute Error (MAE)** was tracked as an additional metric to evaluate model performance. Early stopping was employed to prevent overfitting by monitoring the validation loss. Training automatically stopped when there was no improvement in validation loss for 10 consecutive epochs.

## Training and Validation Loss

Below is the plot showing the training and validation loss curves over epochs:

![Training and Validation Loss](plots/loss_curve.png)

### Insights into the Loss Curve

- **Initial Drop**: Both training and validation loss sharply decrease in the first few epochs, indicating that the model is quickly learning from the data. This initial drop is typical for LSTM models when they capture primary patterns in sequential data.

- **Stabilization**: After a few epochs, both training and validation loss values stabilize, demonstrating that the model has reached a balance between bias and variance. The validation loss remains close to the training loss, which is a good indicator of generalization.

- **Absence of Overfitting**: Thanks to early stopping, the model avoids overfitting as the validation loss does not increase significantly after the initial stabilization. This shows that the model can generalize well to unseen data, which is crucial for real-world applications.

# Model Evaluation Results

Upon evaluation on the test dataset, the model achieved the following metrics:

- **Test Mean Squared Error (MSE)**: 2.6408e-6
- **Test Root Mean Squared Error (RMSE)**: 0.0016
- **R-squared (R²)**: 0.99997

These metrics indicate that the model performs exceptionally well in predicting temperature values, with an extremely low error margin and an R-squared value close to 1, demonstrating near-perfect alignment between predicted and actual values. This impressive performance may be attributed to the comprehensive feature engineering (such as rolling averages and interaction terms), which enables the model to capture nuanced relationships within the battery parameters.

## Actual vs Predicted Temperature

Below is the plot comparing the actual and predicted temperature values across samples. The close alignment of the two curves demonstrates the model's effectiveness in capturing temperature trends.

![Actual vs Predicted Temperature](plots/actual_vs_predicted.png)

**Insights**:
- The **Actual Temperature** (in blue) and **Predicted Temperature** (in red) lines overlap closely, which further confirms the model's high accuracy.
- The periodic fluctuations in temperature are well-captured, showing the model's ability to follow rapid changes and stabilize over varying conditions.
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

| Metric           | Voltage [V]   | Current [A]  | Temperature [°C]  | Capacity [Ah] | WhAccu [Wh] |
|------------------|---------------|--------------|--------------------|---------------|-------------|
| **Count**        | 5,423,272     | 5,423,272    | 5,423,272         | 5,423,272     | 5,423,272   |
| **Mean**         | 351.87        | 3.91         | 8.35              | 0.11          | 0.55        |
| **Std Dev**      | 799.02        | 0.36         | 16.68             | 1.30          | 4.99        |
| **Min**          | 1.00          | 2.26         | -22.50            | -2.78         | -10.30      |
| **25th Percentile** | 9.00      | 3.67         | -9.68             | -0.58         | -2.20       |
| **Median**       | 24.00         | 4.15         | 9.16              | 0.01          | 0.05        |
| **75th Percentile** | 44.00     | 4.19         | 23.83             | 0.94          | 3.69        |
| **Max**          | 6,667.00      | 4.20         | 41.33             | 2.82          | 10.95       |

### Observations:
- **Voltage**: Mean voltage is significantly high, with a wide range from 1V to over 6,667V, indicating potential high-voltage charge cycles.
- **Current**: Current values are consistent, with a mean of around 3.91A and a narrow spread, suggesting controlled discharge/charge cycles.
- **Temperature**: Temperature ranges from -22.5°C to 41.33°C, reflecting testing under a wide range of ambient conditions.
- **Capacity & Accumulated Energy (WhAccu)**: The data includes negative values for Capacity and WhAccu, likely due to discharge cycles, with a maximum capacity observed at 2.82Ah.

This summary provides a snapshot of the battery's operating conditions across various cycles. The broad ranges in voltage and temperature underscore the testing's diversity, which will support a robust temperature prediction model.


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

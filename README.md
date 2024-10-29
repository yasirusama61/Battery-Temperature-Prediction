# Battery Temperature Prediction

This project focuses on predicting the temperature of an LG 18650HG2 battery cell based on various factors such as voltage, current, cycle information, and more. The goal is to develop a robust temperature prediction model that can assist in battery management systems by anticipating potential overheating and ensuring optimal performance.

## Project Overview

Battery temperature is crucial for maintaining the safety and efficiency of lithium-ion cells. This project uses advanced modeling techniques to predict temperature based on time-series data, with unique features engineered to capture cycle-specific behaviors, accumulated energy, and thermal changes.

## Data Description

The dataset includes variables such as:
- **Time Stamp**: Recording time for each entry
- **Cycle**, **Step Time**, **Procedure**: Information about charge/discharge cycles
- **Voltage**, **Current**: Electrical parameters affecting temperature
- **Temperature**: Target variable for prediction

## Goals
1. **Feature Engineering**: Extract cycle-level and interaction features.
2. **Model Development**: Train sequential models like LSTM to predict temperature.
3. **Evaluation**: Assess the model’s accuracy and identify areas for improvement.
4. **Dashboard**: Develop a visualization tool for real-time temperature monitoring.

## Repository Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `src/`: Scripts for data processing and model training.
- `models/`: Saved models.
- `results/`: Plots and model evaluation metrics.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yasirusama61/Battery-Temperature-Prediction.git
   cd Battery-Temperature-Prediction

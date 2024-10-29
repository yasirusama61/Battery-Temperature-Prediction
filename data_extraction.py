"""
Battery Data Extraction Script

Description:
-------------
This script is designed to automate the extraction of relevant columns from multiple CSV files 
within nested folders. It processes data from various ambient temperature folders, 
combines the data, and stores the output in a single Parquet file format for optimized storage.

Output:
--------
The final output is saved as a Parquet file, "extracted_battery_data.parquet", allowing for 
efficient loading and analysis of large datasets.
"""

import os
import pandas as pd

# Define the main dataset folder path
base_folder = "LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020"

# List of folders by temperature
temperature_folders = ["0degC", "10degC", "25degC", "40degC", "n10degC", "n20degC"]

# Define the columns to rename after loading
column_names = [
    'Time Stamp', 'Step', 'Status', 'Prog Time', 'Step Time', 'Cycle', 
    'Cycle Level', 'Procedure', 'Voltage [V]', 'Current [A]', 
    'Temperature [C]', 'Capacity [Ah]', 'WhAccu [Wh]', 'Cnt [Cnt]', 'Unnamed: 14'
]

# Define the specific columns to extract
columns_to_extract = ['Time Stamp', 'Step', 'Status', 'Voltage [V]', 'Current [A]', 
                      'Temperature [C]', 'Capacity [Ah]', 'WhAccu [Wh]']

# Initialize an empty DataFrame to hold all the extracted data
all_data = pd.DataFrame()

# Loop through each temperature folder
for folder in temperature_folders:
    folder_path = os.path.join(base_folder, folder)
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Folder {folder_path} does not exist. Skipping...")
        continue

    print(f"Processing folder: {folder_path}")  # Debugging statement

    # Loop through each CSV file in the current folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            print(f"Processing file: {file_path}")  # Debugging statement
            try:
                # Read the CSV file, skipping initial rows and renaming columns
                data = pd.read_csv(file_path, skiprows=28)  # Header starts on row 29 (0-based index 28)
                data = data.drop(0)  # Drop the row containing units
                
                # Rename columns to maintain consistency
                data.columns = column_names
                
                # Extract the relevant columns
                data_filtered = data[columns_to_extract].copy()
                
                # Add ambient temperature based on folder name
                if "degC" in folder:
                    temperature_label = folder.replace("degC", "°C").replace("n", "-")
                    data_filtered["Ambient Temperature"] = temperature_label
                    print(f"Assigned Ambient Temperature: {temperature_label}")  # Debugging statement
                else:
                    data_filtered["Ambient Temperature"] = "Unknown"
                
                # Ensure all selected columns are numeric where applicable, coercing errors to NaN
                data_filtered['Voltage [V]'] = pd.to_numeric(data_filtered['Voltage [V]'], errors='coerce')
                data_filtered['Current [A]'] = pd.to_numeric(data_filtered['Current [A]'], errors='coerce')
                data_filtered['Temperature [C]'] = pd.to_numeric(data_filtered['Temperature [C]'], errors='coerce')
                data_filtered['Capacity [Ah]'] = pd.to_numeric(data_filtered['Capacity [Ah]'], errors='coerce')
                data_filtered['WhAccu [Wh]'] = pd.to_numeric(data_filtered['WhAccu [Wh]'], errors='coerce')
                
                # Append the filtered data to the main DataFrame
                all_data = pd.concat([all_data, data_filtered], ignore_index=True)
                print(f"Rows in all_data after processing {file_path}: {all_data.shape[0]}")  # Check row count

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Save the combined data to a Parquet file for further analysis
output_file = "extracted_battery_data.parquet"
all_data.to_parquet(output_file, index=False)
print(f"Data extraction complete. Combined data saved to {output_file}. Rows in final Parquet file: {all_data.shape[0]}")

# PyBaMM Battery Simulation Script
# Author: Usama Yasir Khan
# Description: This script runs a battery simulation using PyBaMM for a specified experiment and saves the results as a CSV file.

import pybamm
import pandas as pd
import numpy as np

def run_simulation_and_save(experiment, parameter_values, filename, nominal_capacity=5):
    """
    Run the battery simulation using PyBaMM for the given experiment and save the results as a CSV.

    Parameters:
    - experiment: PyBaMM experiment object for running the simulation.
    - parameter_values: PyBaMM ParameterValues object.
    - filename: Path to the CSV file where the results will be saved.
    - nominal_capacity: Nominal capacity of the battery in Ah (default: 5 Ah).
    """
    # Load a pre-built model with thermal effects (DFN model with thermal considerations)
    model = pybamm.lithium_ion.DFN({"thermal": "x-lumped", "cell geometry": "pouch"})

    # Define a fluctuating ambient temperature function
    def ambient_temperature_function(t, T, y=None):
        # 298.15 K is 25°C, and the temperature fluctuates by ±10 K over a period
        return 298.15 + 10 * np.sin(0.001 * t)

    # Set the function for ambient temperature in parameter values
    parameter_values.update({
        "Ambient temperature [K]": ambient_temperature_function,
        "Initial temperature [K]": 298.15,  # Start at 25°C
        "Total heat transfer coefficient [W.m-2.K-1]": 5  # Adjust cooling coefficient
    })

    # Create a simulation with the specified experiment and parameter values
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)

    # Run the simulation
    solution = sim.solve()

    # Extract time, voltage, current, and temperature from the solution
    time_data = solution["Time [s]"].entries
    voltage = solution["Voltage [V]"].entries
    current = solution["Current [A]"].entries
    temperature = solution["X-averaged cell temperature [K]"].entries

    # Convert temperature from Kelvin to Celsius
    temperature_c = temperature - 273.15

    # Calculate cumulative capacity (Ah) from the current
    delta_time_hours = np.diff(time_data / 3600, prepend=0)  # Time in hours
    cumulative_capacity = np.cumsum(current * delta_time_hours)

    # Calculate SOC
    soc = 1 - (cumulative_capacity / nominal_capacity)  # SOC is in fraction

    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'Time [s]': time_data,
        'Voltage [V]': voltage,
        'Current [A]': current,
        'SOC': soc,
        'Temperature [C]': temperature_c
    })

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Simulation results saved to {filename}")

# Define the experiment configuration
experiment = pybamm.Experiment([
    "Discharge at 0.4C until 2.5 V",
    "Rest for 10 minutes",
    "Charge at 0.5C until 4.2 V",
    "Hold at 4.2 V until 50 mA",
    "Rest for 10 minutes"
] * 10,  # Repeat this for 100 cycles
period="1 second")

# Load the parameter values (use "Chen2020" or another predefined set)
parameter_values = pybamm.ParameterValues("Chen2020")

# Run the simulation and save the results to a CSV (assuming a 5 Ah nominal battery capacity)
run_simulation_and_save(experiment, parameter_values, "battery_simulation_with_fluctuations.csv", nominal_capacity=5)

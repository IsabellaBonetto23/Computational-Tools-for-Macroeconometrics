# Computational-Tools-for-Macroeconometrics
# Computational Tools for MacroEconometrics – Forecasting Assignment

This repository contains the code for a forecasting assignment using the FRED-MD dataset. The assignment demonstrates data cleaning, transformation, ARX model construction, model estimation using Ordinary Least Squares (OLS), forecast simulation, real-time evaluation, and visualization of forecast errors and forecasts.

## Overview

In this assignment, we focus on forecasting the industrial production (INDPRO) using an ARX model with lagged values of INDPRO as well as exogenous variables (CPIAUCSL and TB3MS). The project is organized as follows:

1. **Data Loading and Cleaning:**  
   - Downloads and loads the FRED-MD dataset.
   - Drops unnecessary rows (such as transformation codes).
   - Resets indexing and converts date columns into proper datetime format.

2. **Transformation of Variables:**  
   - Extracts transformation codes from the dataset.
   - Defines and applies a function to transform variables (e.g., first difference, log transformation) based on the codes.

3. **Data Visualization:**  
   - Plots key economic indicators (INDPRO, CPIAUCSL, TB3MS) with proper formatting and labels.

4. **ARX Model Construction:**  
   - Constructs the matrix of lagged variables for the target (INDPRO) and predictor variables.
   - Adds a constant column for the intercept.

5. **Model Estimation and Forecasting:**  
   - Estimates model parameters using OLS.
   - Simulates one-step ahead forecasting and rolls the predictor vector to generate forecasts for multiple periods.

6. **Model Evaluation:**  
   - Computes error metrics (MSE, RMSE, MAE) to evaluate forecast accuracy.
   - Implements a real-time evaluation simulation to assess forecast performance over multiple cutoff dates.
   
7. **Forecast Visualization:**  
   - Produces graphs comparing actual data against forecasts.
   - Displays the evolution of forecast errors over time.

## How to Run

1. **CLONE THE REPOSITORY:**
bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
2. INSTALL DEPENDENCIES:
Ensure you have Python 3.10 installed. Then install the required libraries using pip:

pip install pandas numpy matplotlib scikit-learn
3. RUN THE SCRIPT:
You can run the entire script in an interactive environment like Jupyter Notebook or as a standalone Python script:

python your_script_name.py
Running the entire script (rather than piece by piece) ensures that all variables are defined correctly and that the graphs are produced as expected.
File Structure

README.md: This file.
assignment1.1.py: The main Python script containing the full assignment code.
current.csv: The FRED-MD dataset used in the analysis.
Output

When you run the script, you will see:

Console output showing dataset dimensions, error metrics (MSE, RMSE, MAE), and RMSFE evaluation.
Several graphs:
A comparison of actual data and forecasts over a specified period.
The evolution of forecast errors over time.
These outputs help evaluate the forecasting model’s performance and illustrate the economic trends captured by the ARX model.

ECONOMIC AND ECONOMETRIC BACKGROUND:

This assignment is designed to demonstrate the process of transforming economic time series data to prepare it for forecasting. Key steps include:

- Data Transformation: Removing trends and stabilizing variance using techniques such as differencing and logarithmic transformations.
- Lagged Variables: Using past values of the target and predictor variables (ARX model) to forecast future values.
- Real-Time Evaluation: Simulating a forecaster who updates predictions as new data becomes available, and evaluating performance using metrics like MSFE and RMSFE.
Understanding these steps is essential for building robust economic forecasting models that can inform policy decisions and business strategies.

---


This README provides an overview of the assignment, instructions for running the code, a description of each major section, and a brief discussion of the economic and econometric rationale behind the analysis.























# We import libraries into the program
import pandas as pd # For managing data in neat tables (DataFrame)
import numpy as np # For advanced math calculations and useful for the arrays we will need
import matplotlib.pyplot as plt  # For data visualization
import matplotlib.dates as mdates  # For managing dates in plots

# ------------------------------- #
# DOWNLOADING THE FRED-MD DATASET 
# ------------------------------- #

# It's the first step for our econometric analysis because we need data on economic indicators to build the model; in this way we can clean it, analyze it and make forecasts.
file_path = "/Users/isabellabonetto/Desktop/macroeconometrics_lab1.1/current.csv" # file_path → stores the location of our CSV file
# Then we load the dataset from local file
df = pd.read_csv(file_path) # This reads the CSV file and loads the data into a DataFrame called df.

# ------------------------------- #
# DATAFRAME CLEANING 
# ------------------------------- #

# We start with raw data that might contain "extra" information we don't need for analysis. The first row contains transformation codes of variables rather than actual data, so we remove it
df_cleaned = df.drop(index=0) # This part assigns the result of the dropping operation to a 🆕 variable called df_cleaned → more ✅accurate and containing only ✅relevant information!

# Time series analysis requires us to properly reset the row numbering (indexing) after dropping the first row, so that df_cleaned starts at 0 again, without gaps
df_cleaned.reset_index(drop=True, inplace=True) # drop=True → tells Python not to keep the old row numbers as a new column, to avoid confusion in the data 🚫 ; inplace=True → everything gets updated 🔄 in place in the same DataFrame, so that code saves memory and improves efficiency.

# In order to build a reliable model, we proceed converting the 'sasdate' column to datetime format for easier handling
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y') # df_cleaned → selects the column named 'sasdate' (containing the dates of each observation) from our DataFrame called df_cleaned ; pd.to_datetime(...) → it's a function from the pandas library that converts a series (or column) of text (strings) into proper date objects that Python can understand as dates (standardized format📅) ; format='%m/%d/%Y' → converts the sasdate column into month/day/year format. 
# "=" → this operator assigns the converted dates back to the 'sasdate' column in df_cleaned 📊

# We verify that the first few rows present data correctly loaded
print(df_cleaned.head()) 

# Dataset imported successfully✅. 
# CHECKING THE DATA: we can make additional checks in order to print out parts of the DataFrame and verify that data are correctly imported and cleaned
print(df_cleaned.shape)  # It verifies the total number of rows and columns: we get exactly 792 rows and 127 columns.
print(df_cleaned.columns) # We control column names
print(df_cleaned.dtypes) # We control the right format: datetime for sasdate and float64 for numerical data.
print(f"Total number of rows: {df_cleaned.shape[0]}") # .shape → returns a pair of numbers that tell the size of the DataFrame: ( 792 = number of rows, 127 = number of columns); [0] → selects the first number from the pair provided by .shape, which is the number of rows (792)


# ------------------------------- #
# TRANSFORMATION OF VARIABLES #
# ------------------------------- #

# Let's extract the transformation codes from the first line
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# We want to correctly match each economic variable with its corresponding transformation.
# df.iloc[0, 1:] → Selects the first row (index 0) that contains the transformation codes for how each variable should be changed before analysis, excluding the first column (sasdate). This row contains the transformation codes.
# .to_frame() → Converts the selected row (which is initially a Series-- a single column of data) into a DataFrame (a table with rows and columns).
# .reset_index() → Resets the index of the new DataFrame so that the rows are numbered in order, starting from 0.
# transformation_codes.columns = ['Series', 'Transformation_Code'] → Renames the columns of the DataFrame to be 'Series' and 'Transformation_Code'.

# So, we have a DataFrame called transformation_codes with two columns:
#Series → The names of the economic variables.
#Transformation_Code → The transformation code that needs to be applied to each variable.

# To show this new dataframe:  
print(transformation_codes) #As we can see in this new dataframe we have 126 rows (because sasdate is exluded) and 2 columns

#WHAT ARE TRANSFORMATION CODES AND WHY DO WE NEED THEM IN OUR ANALYSIS❓

# 🟠Code 1: No transformation → The variable is already stationary, so no changes are needed. This means that the variable does not have a trend or changing variance over time and it's ready to use.
# Typically: short-term interest rates, unemployment rates (often stationary in raw form).

# 🟠Code 2: First difference (ΔX_t = X_t - X_{t-1}) → Used for non-stationary series, the previous value is subtracted from the current one to remove steady upward or downward linear trends (like a random walk).
# Typically: GDP, stock prices, money supply.

# 🟠Code 3: Second difference (Δ²X_t = ΔX_t - ΔX_{t-1}) → This can be useful when a single difference isn't enough to remove the trend, especially if the trend is curved (quadratic) and requires a second differencing step.
# Typically: some GDP series or industrial production indices.

# 🟠Code 4: Log transformation (log(X_t)). The natural logarithm of the data helps stabilize variance, especially when data grows exponentially. 
# Typically: Stock market indices, CPI, or real GDP data with exponential growth and high heteroskedasticity.

# 🟠Code 5: First difference of log (Δlog(X_t) = log(X_t) - log(X_{t-1})) → The difference of logs is similar to calculating the percentage change of the variable. Used when a variable follows an exponential trend and the interest is in capturing the growth rate.
# Typically: GDP growth, inflation rate, productivity growth.

# 🟠Code 6: Second difference of log (Δ²log(X_t)) → Used for series with strong exponential trends.
# It applies a second differencing after the first difference log transformation.
# Typically: Inflation acceleration, productivity growth rate, where the trend is highly non-linear.

# 🟠Code 7: Approximate percentage change (Δ((X_t / X_{t-1}) - 1)) → Instead of taking logs, this method calculates the relative change between consecutive periods.
# It's a valid alternative to capture the growth rate compared to logarithms.
# Typically: changes in the inflation rate or exchange rate fluctuations.

# Let's start a function called apply_transformation that automates correctly the transformation of data considering series (a column of data) and code 
def apply_transformation(series, code):
    if code == 1:
        return series  # No transformation
    elif code == 2:
        return series.diff()  # First difference
    elif code == 3:
        return series.diff().diff()  # Second difference
    elif code == 4:
        return np.log(series)  # Log transformation
    elif code == 5:
        return np.log(series).diff()  # First difference of log
    elif code == 6:
        return np.log(series).diff().diff()  # Second difference of log
    elif code == 7:
        return series.pct_change()  # Approximate percentage change
    else:
        raise ValueError("Invalid transformation code")

#If a variable has code 1, we simply return the original data without changing it; if it has code 2, then the first difference is computed and so on.
#If the code doesn’t match any of the known recipes (1 through 7), the function stops and gives an error message 🚫❓

for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))
# This loop goes through each variable (series) and its corresponding transformation code, converts the variable to a float (number), and applies the transformation.
 
# Many of the transformations introduce missing values (NaN) at the beginning of the time series.

# FURTHER CLEANING: 

df_cleaned = df_cleaned[2:]  # It removes the first two rows that now contain missing values (because of the transformations) from df. 

# After removing rows, the index will still refer to the original row numbers

df_cleaned.reset_index(drop=True, inplace=True) # reset_index(drop=True, inplace=True):
# drop=True → prevents the old index from being added as a new column.
# inplace=True → modifies the DataFrame directly (NO DUPLICATION needed)

# To check the transformed data 
df_cleaned.head() # Indeed, running this command, we can easily notice that the 0th observation starts from March 1959 (and not anymore from January 1959)


# ------------------------------- #
# DATA VISUALIZATION #
# ------------------------------- #
# Now we want to create graphs for chosen key economic indicators in our dataframe


# Let's consider three economic series to visualize (INDPRO, CPIAUCSL, TB3MS) and assign them human-readable names (“Industrial Production”, “Inflation (CPI)”, “3-month Treasury Bill rate”).
series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS'] 
series_names = ['Industrial Production', 'Inflation (CPI)', '3-month Treasury Bill rate'] 
# Note that the order of the lists matches so that each name corresponds to its variable.

# Creating a plot with multiple subplots (one for each selected series)
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))

# plt.subplots → is a function from Matplotlib that sets up a grid of subplots 
# len(series_to_plot) → it yields the number of subplots we need (since we have 3 items, it creates 3 subplots).
# 1 → the subplots are arranged in one column (stacked vertically).
# figsize=(8, 15) → it specifies the size of the overall figure (8 inches wide and 15 inches tall).

# We can ensure that axs (the array of subplot axes) is always treated as an array, even if there's only one subplot; this simplifies the code later on when we should loop through each subplot to add data and labels.
axs = np.atleast_1d(axs)
# The below line prints out the number of items in series_to_plot and series_names 🔢✅
print(len(series_to_plot), len(series_names))  

# We now have to run the whole following command from 1️⃣ to 1️⃣0️⃣: 
                     
                                                                             
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):   #1️⃣ Looping over Subplots and Variables and checking if the series exists. zip(axs, series_to_plot, series_names) → This line loops simultaneously over the subplots (ax), dataset columns (series_name), and their friendly names (plot_title).                                                                           
    if series_name in df_cleaned.columns:                                    # if series_name in df_cleaned.columns: → This checks if the current series (variable) exists in our cleaned data series (df_cleaned DataFrame columns).
        
                                                                             #2️⃣ Converting dates. This converts the sasdate column to proper date format so that the x-axis (time) is recognized correctly. 
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')     # It ensures that observations are correctly ordered and that trends over time are properly displayed.
        
                                                                             #3️⃣ Plotting the Series. This plots the data for the current series against the dates on the current subplot. The label parameter uses the friendly name. 
        ax.plot(dates, df_cleaned[series_name], label=plot_title)            # The general command is ax.plot(x_values, y_values, label='Legend Name') Where: 
                                                                             # x_values = dates (time on the x-axis).
                                                                             # y_values = df_cleaned[series_name] (economic indicator on the y-axis).
                                                                             # label=plot_title → adding the legend with human-readable name.
        
                                                                             #4️⃣ Setting and Formatting X-Axis Ticks. This sets major ticks on the x-axis every 5 years. 
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))               # Properly spaced ticks help in accurately interpreting long-term trends in economic data, making the graph easier to read.    
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))             # This formats the x-axis tick labels to show only the year (e.g., "2000")

                                                                             #5️⃣ Setting the Plot Title. This adds a title to the subplot using the friendly name.
        ax.set_title(plot_title)                                             # A clear title viewers quickly understand which economic indicator is being visualized (effectiveness of communication)
        
                                                                             #6️⃣ Labeling the Axes. We label the x-axis “Year,” and the y-axis “Transformed Value,” to indicate that the data was transformed before plotting.
        ax.set_xlabel('Year')                                                # This comand labels the x-axis as "Year."
        ax.set_ylabel('Transformed Value')                                   # This comand labels the y-axis to indicate that the values were transformed before plotting.

                                                                             #7️⃣ Adding a legend.
        ax.legend(loc='upper left')                                          # This places a legend in the upper left of each subplot. It helps differentiate between multiple data series in a plot.

                                                                             #8️⃣ Rotating X-Axis Labels. We rotate the x-axis labels by 45 degrees and aligns them to the right. Rotated labels improve legibility, especially when there are many tick marks. 
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')    # This command rotates the year labels 45 degrees to prevent overlap. (ha='right') aligns the labels to the right for better readability.

    else:
        ax.set_visible(False)                                                #9️⃣ Handling Missing Data. If the current series is not found in the DataFrame, the corresponding subplot is hidden → this ensures that only available, relevant economic indicators are displayed and prevents empty confusionary plots.



#1️⃣0️⃣ Adjusting the Layout. plt.tight_layout() → automatically adjusts spacing between suplots to avoid overlap and allowing a comprehensive analysis.
plt.tight_layout()

#1️⃣1️⃣ Displaying the Figure. plt.show() → displays the final figure helping detect trens and understand the underlying economic condition (size is 640x480 pixels)
plt.show()

# ------------------------------- #
# ARX MODEL CONSTRUCTION #
# ------------------------------- #

# Selecting the target variable that we want to predict later (Industrial Production) from our cleaned table (called df_cleaned) and naming it Yraw
Yraw = df_cleaned['INDPRO'] ## Industrial production is a key indicator of how well the economy is doing

# Selecting target and predictor variables (Inflation and 3-month Treasury Bill Rate which is a short term interest rate)
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']] ## These are our predictor (or explanatory) variables. In an ARX model, they help explain changes in the target variable (industrial production)

# Configuring the number of lags and the forecasting horizon (lead)
num_lags = 4  # p = number of lags taken into account: we only look at the last 4 periods for our variables
num_leads = 1 # h = 1-step ahead prediction 

# Creating the matrix of independent variables (X)
X = pd.DataFrame() # This line creates an empty table called X where we will store all the shifted (lagged) values.
for lag in range(0, num_lags + 1): # This loop tells the computer to do something repeatedly for each value of lag starting from 0 up to num_lags (= 4), so it creates several columns, one for each past period: 0, 1, 2, 3, 4. The range function generates a sequence of numbers, starting from 0 and not including num_lags + 1
    X[f'INDPRO_lag{lag}'] = Yraw.shift(lag) # X[f'INDPRO_lag{lag}'] → creates a new column in our DataFrame X (if lag is 0, it becomes 'INDPRO_lag0',...,'INDPRO_lag4'), we need to store each shifted version of our industrial production series (Yraw) in a separate column so that our model knows which value comes from which period ; = Yraw.shift(lag) → this assigns a value to our new column and moves the data by the number of periods indicated by lag. This lagged series version has a crucial role in the AR component
for col in Xraw.columns: # For every column in Xraw (containing CPIAUCSL and TB3MS)
    for lag in range(0, num_lags + 1): # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag) # X[f'{col}_lag{lag}']→ it creates a new column name from Xraw (for example, the inflation values); Xraw[col].shift(lag) -> this shifts the column by the number indicated by lag. The creation of these predictor lagged variables (the X part) makes the model an ARX model. This means the future value of industrial production might depend not only on its own past values but also on past values of inflation and interest rates



# Adding a column of ones at the very beginning of the table X in order to estimate the intercept, which is the constant term in the regression model, the part of the prediction that is not explained by the other variables/lagged values (it acts as a baseline )
X.insert(0, 'Ones', np.ones(len(X)))  
# np.ones(len(X))) → this creates a NumPy array filled with ones. The length of the array is equal to the number of rows in X, so that each observation in your model gets its own intercept term

# X is now a DataFrame
X.head() # We can notice that the first p = 4 rows of X have missing values (NaN)

# Dropping missing values (NaN) and cleaning the dataset: SETTING THE STAGE FOR OLS REGRESSION

X_T = X.iloc[-1:].values # X.iloc[-1:].values → this selects the last row of X and converts it into an array; this last row called X_T will be used later to make the 1-step ahead prediction

# Creating the vector y 
y = Yraw.shift(-num_leads).iloc[num_lags:-num_leads].values
y 
# Yraw.shift(-num_leads) → moves all the numbers to the left by a certain number of positions, creating a new series that is "ahead" of the original and it's used as our "target" (what we want to predict)
# .iloc[num_lags:-num_leads] → after shifting, we use .iloc to select only a part of our series; num_lags tells us to start at a certain position (from p+1); -num_leads tells us to stop before (up to h-1) because there is no "next" value after the last period T (indeed we can notice that y has missing values in the last h positions)
# .values → this converts the selected part of the series into a plain NumPy array for easier math operations later

X = X.iloc[num_lags:-num_leads].values # Subset getting only rows of X from p+1 to h-1

X_T

# ------------------------------- #
# MODEL ESTIMATION AND FORECASTING #
# ------------------------------- #

from numpy.linalg import solve # This line imports the function called solve from the NumPy library’s linear algebra module: in our model, we need to solve an equation to get the best-fitting numbers (coefficients) for our regression. This is a key part of estimating our model using Ordinary Least Squares (OLS)

# Estimating model parameters using the Ordinary Least Squares (OLS) method: 
beta_ols = solve(X.T @ X, X.T @ y) # This line calculates the coefficients of our forecasting model solving for the OLS estimator beta: (X'X)^{-1} X'Y; where X.T means the transpose of matrix X (flipping rows and columns), @ is the operator for matrix multiplication and solve(A, B) finds the solution β in the equation Aβ = B

# Making a one-step-ahead prediction
forecast = X_T @ beta_ols * 100 # X_T @ beta_ols multiplies the most recent row of data (X_T) by our coefficients (beta_ols) to get a forecasted value; * 100 gives the forecast in percentage points → we predict the next period's industrial production using the current available information
print(f"One-step ahead forecast: {forecast[0]:.2f}%") # Shows the forecasted value on the screen
forecast # This variable contains now the one-step ahead (h = 1 forecast in this case) of INDPRO expressed in percentage change. Showing this forecast allows us to later compare our predicted value with the actual value, evaluate the model's performance, and ultimately make informed economic decisions.
# Essentially we predicted what industrial production would be for the next time period, and our model’s guess came out as approximately 0.39%. This value is then used to calculate errors and evaluate the model's overall forecasting performance over multiple periods.

# ASSESSING QUALITY: we are now interested in knowing if our predictions (forecasts) are close to the real values. In this forecasting exercise, we want to see how good our model is at predicting future values of industrial production (INDPRO). However, rather than waiting month by month to see if our predictions are correct, we simulate (or pretend) we’re making predictions repeatedly using past historical data. This is called a Real-time evaluation.

def calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = '12/1/1999', target = 'INDPRO', xvars = ['CPIAUCSL', 'TB3MS']): # This defines a function named calculate_forecast that takes several parameters: ✽ df_cleaned: our cleaned dataset ; ✽ p: number of lags (default is 4, so we are using the last 4 observations available at the cutoff date) ; ✽ H: A list of forecast horizons (1, 4, and 8 months ahead, for example January, April and August 2000 respectively) ; ✽ end_date: the cutoff (limit) date for the data ; ✽ target: the target variable (here, 'INDPRO') ; ✽ xvars: a list of predictor variables
    ## Subset df_cleaned to use only data up to the given end_date 
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)] # It creates a new DataFrame (rt_df) that contains only the data up to the specified end_date; since we are putting ourselves in the shoes of a forecaster who has been using the forecasting model for a long time, we are pretending it’s December 1999 and that we don’t know what happens later.
    
    ## Get the actual values of target at different steps ahead
    Y_actual = [] # Initializes an empty list Y_actual: we need to collect the real (or "actual") values of our target variable (like industrial production) for the days we forecast. These actual values are used later to calculate how far off our predictions were (forecast errors)
    for h in H: # This starts a loop that goes through each value in H. Here, H is a list of forecast horizons (for example, H = [1, 4, 8]), which means we want to forecast 1 month ahead, 4 months ahead, and 8 months ahead. The quality of forecasts can vary depending on how far into the future you try to predict.
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h) # ✽ pd.Timestamp(end_date) → converts the given end date (like '12/1/1999') into a date object; ✽ pd.DateOffset(months=h) creates an offset of h months ►►► Adding these together gives a new date, os, which is h months after the end date. This new date os represents the future time point for which we want the actual value (it tells us the date we need to compare our forecast with the real data)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100) # ✽ df_cleaned[df_cleaned['sasdate'] == os] → selects the row in the cleaned DataFrame where the date is equal to os (the future date we calculated); ✽ [target] picks the column for our target variable (like 'INDPRO') ; ✽ * 100 multiplies the value by 100 ; ✽ .append(...) adds this value to our Y_actual list
    ## Now Y contains the true values at T+H (multiplying * 100). Later, these will be compared to the forecasts produced by our model to assess accuracy using metrics like the Mean Squared Forecast Error (MSFE).
    
    ## Defining Yraw and Xraw: 
    Yraw = rt_df[target] # Yraw becomes the target variable from the real-time DataFrame.
    Xraw = rt_df[xvars] # Xraw becomes the predictor variables from that same DataFrame.
    # We use these subsets to build our regression model with only the available data up to the cutoff date.
    
    ## Building the Lagged Matrix X
    X = pd.DataFrame() # This line creates an empty table (called a DataFrame) named X → we need to store all the new columns that show the past values (lags) of our variables
    ## Add the lagged values of Y
    for lag in range(0, p): # This loop goes through a list of numbers from 0 up to p-1 (p = 4, so it goes 0, 1, 2, 3)
    # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{target}_lag{lag}'] = Yraw.shift(lag) # This line creates a new column in X with a name like "INDPRO_lag0", "INDPRO_lag1", etc. The function Yraw.shift(lag) moves the target variable values down by the number given by lag. This creates a record of the past values of the target variable that the model will use to help predict the future.
    ## Adding Lagged Values of the Predictor Variables
    for col in Xraw.columns: # This loop goes through each column in Xraw. Indeed, we want to create lagged versions of each predictor variable, not just the target because also past values of Inflation and Treasury Bill rate may influence industrial production.
        for lag in range(0, p): # For each predictor variable, this inner loop goes through each lag value from 0 to p - 1. In this way we ensure a full set of past values for every predictor variable.
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag) # This creates a new column for the exogenous predictor variables (say, CPIAUCSL) with a name like "CPIAUCSL_lag0", "CPIAUCSL_lag1", etc. It shifts the column’s values by the specified lag.
    
    ## Add a column on ones (for the intercept)
    X.insert(0, 'Ones', np.ones(len(X))) # This inserts a column named "Ones" at the very beginning of X, where every entry is the number 1.
    
    ## Save last row of X (converted to numpy)
    X_T = X.iloc[-1:].values # It selects the last row of X (which contains the most recent lagged values) and converts it to a NumPy array. This row will be used to generate the forecast because it represents the most current information available, which is used for making the one-step-ahead forecast.
    
    ## While the X will be the same, Y needs to be leaded differently
    Yhat = [] # This creates an empty list called Yhat in order to store the forecast for each different time horizon. These collected forecasts will later be compared with the actual values to assess how well our model is performing.
    for h in H: # It starts a loop that goes through each forecast horizon in the list H.
        y_h = Yraw.shift(-h) # This line creates a new series called y_h by shifting Yraw to the left by h periods.
        ## Subset getting only rows of X and y from p+1 to h-1
        y = y_h.iloc[p:-h].values # This line selects a portion of the shifted series y_h. It uses .iloc[p:-h] to choose only the rows where we have complete information for both predictors and the future value. Then it converts that part into a NumPy array using .values. The correct alignment of the dependent variable with the predictor variables is crucial. This step ensures that the model is estimated on the same set of complete data points (removing the first p rows and the last h rows because of missing values), avoiding biases due to missing data.
        X_ = X.iloc[p:-h].values  # Similarly, this line selects the corresponding rows from the lagged predictor matrix X (from row p to row -h) and converts them into a NumPy array.
        ## This ALIGNMENT is critical for producing unbiased and efficient estimates of the model coefficients using OLS
        # Solving for the OLS estimator beta: (X'X)^{-1} X'Y
        beta_ols = solve(X_.T @ X_, X_.T @ y) # This line calculates the OLS coefficients (beta values) by solving the normal equations: beta = (X'X)^{-1} X'Y. This step finds the weights for each lag.
        # The OLS method minimizes the difference between the actual values and the predictions by finding the best-fitting coefficients for the predictors → it tells us how strongly each past observation (both of the target and the predictors) affects the future value.
        
    ## Produce the One step ahead forecast
        ## % change month-to-month INDPRO
        Yhat.append(X_T @ beta_ols * 100) # Recalling that X_T is the last row of the full predictor matrix (the most recent available data), X_T @ beta_ols multiplies this row with the estimated coefficients to produce a forecast. The forecast is converted into percentage points. 
    
    ## Now calculate the forecasting error and return
    return np.array(Y_actual) - np.array(Yhat) # Through np.array(Y_actual) we convert our list of actual values, Y_actual, into a well structured NumPy array, making easier to compare these actual values with our model's predictions. Using np.array(Yhat), the list of forecasts, Yhat, is converted into a NumPy array; in this way it's possible to make a direct comparison.
    # ACCURACY OF THE MODEL: subtracting gives us the forecast error—that is, the difference between what actually happened and what our model predicted. A smaller error means our model’s predictions are close to the real values, which is good. A larger error would mean our model is not predicting well. 
    # With the return function we can calculate real-time errors by looping over the end_date to ensure we end the loop at the right time. By returning the forecast errors, we can later compute metrics like MSFE or RMSFE.


t0 = pd.Timestamp('12/1/1999') # This line creates a timestamp (a special kind of date object) that represents December 1, 1999
e = [] # This creates an empty list called e to store all the forecast errors from different periods. These errors help us evaluate the overall forecasting performance
T = [] # This creates an empty list called T to keep track of the cutoff dates and later see how the forecasting errors evolve over time
for j in range(0, 10): # This line starts a loop that will run 10 times. The loop simulates the process of making repeated forecasts over time (real-time evaluation). Each iteration represents moving forward one month and making new forecasts.
    t0 = t0 + pd.DateOffset(months=1) # This updates the date t0 by adding 1 month to it. This way, we repeatedly forecast as if we are in the past.
    print(f'Using data up to {t0}') # As we've already seen, this line prints a message on the screen showing the current cutoff date. Printing the cutoff date helps in tracking the forecasting process and verifying that the simulation is running as expected.
    ehat = calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = t0) # This line calls the function calculate_forecast with the current cutoff date t0 and other parameters (4 lags, and forecast horizons of 1, 4, and 8 months). It stores the forecast errors in the variable ehat.
    e.append(ehat.flatten()) # This line adds the forecast errors from ehat (flattened into a simple list) into the basket e.
    T.append(t0) # This line adds the current cutoff date t0 into T

    ## Create a pandas DataFrame from the list
edf = pd.DataFrame(e) # This converts the list e (which holds forecast errors) into a pandas DataFrame called edf. Having forecast errors in a DataFrame format makes it convenient to apply statistical calculations (averages or variances).
    ## Calculate the RMSFE (Root Mean Squared Forecast Error), that is, the square root of the MSFE
np.sqrt(edf.apply(np.square).mean()) # edf.apply(np.square) squares each error in the DataFrame; .mean() calculates the average of these squared errors, giving the Mean Squared Forecast Error (MSFE); np.sqrt(...) takes the square root of the MSFE to produce the RMSFE, that is a common metric used to assess forecast accuracy. A lower RMSFE indicates that the forecasts are generally close to the actual values, meaning the model is performing well.


# Forecast for the next 12 months
num_predictions = 12
predictions = []
for _ in range(num_predictions):
    forecast = X_T @ beta_ols
    predictions.append(forecast[0])
    X_T = np.roll(X_T, shift=-1)
    X_T[-1] = forecast[0]

# print(f"📈 Forecasts for the next {num_predictions} months:")
# print(predictions)

# ------------------------------- #
# MODEL EVALUATION #
# ------------------------------- #

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Compute errors
y_pred = X @ beta_ols
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")


# ------------------------------- #
# FILTERED DATASET AND FORECAST EVALUATION #
# ------------------------------- #

# Filter data for the period 2020-2025
df_filtered = df_cleaned[(df_cleaned['sasdate'] >= '2000-01-01') & (df_cleaned['sasdate'] <= '2025-12-31')]

# Select target and predictor variables
y = df_filtered['INDPRO']
y_pred = y.shift(-1)  # Simulated forecast (you can replace this with the ARX model)

# Compute errors
mse = mean_squared_error(y[:-1], y_pred.dropna())
rmse = np.sqrt(mse)
mae = mean_absolute_error(y[:-1], y_pred.dropna())

# Improved error printout
print("============================")
print("      ERROR EVALUATION 2000-2025     ")
print("============================")
print(f"MSE  (Mean Squared Error)  : {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE  (Mean Absolute Error) : {mae:.4f}")
print("============================")

# ------------------------------- #
# FILTERED FORECAST VISUALIZATION #
# ------------------------------- #

plt.figure(figsize=(12,6))
plt.plot(df_filtered['sasdate'], y, label="Actual Data", color="blue")
plt.plot(df_filtered['sasdate'], y_pred, label="Forecast", color="red", linestyle="dashed")

plt.xlabel("Date", fontsize=12)
plt.ylabel("Industrial Production", fontsize=12)
plt.title("Comparison Between Actual Data and Forecasts (2000-2025)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show(block=False)

# ------------------------------- #
# FUNCTION TO CALCULATE FORECASTS #
# ------------------------------- #

def calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date='12/1/1999', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    # Subset dataset up to the end date
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    
    # Actual target values at different forward steps
    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target] * 100)

    Yraw = rt_df[target]
    Xraw = rt_df[xvars]
    X = pd.DataFrame()
    
    # Add lags of the target variable
    for lag in range(0, p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)
    
    # Add lags of explanatory variables
    for col in Xraw.columns:
        for lag in range(0, p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
    
    # Add a column of 1s for the intercept
    X.insert(0, 'Ones', np.ones(len(X)))
    
    # Save the last row of X (converted to numpy)
    X_T = X.iloc[-1:].values
    
    # Compute forecasts at different horizons
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        beta_ols = solve(X_.T @ X_, X_.T @ y)  # OLS estimation
        Yhat.append(X_T @ beta_ols * 100)  # One-step ahead forecast
    
    # Compute forecast error
    return np.array(Y_actual), np.array(Yhat), np.array(Y_actual) - np.array(Yhat)

# Simulate forecasts for multiple dates
t0 = pd.Timestamp('12/1/1999')
e = []
T = []
Y_actuals = []
Y_forecasts = []

for j in range(0, 10):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    Y_actual, Y_forecast, ehat = calculate_forecast(df_cleaned, p=4, H=[1, 4, 8], end_date=t0)
    e.append(ehat.flatten())
    Y_actuals.append(Y_actual.flatten())
    Y_forecasts.append(Y_forecast.flatten())
    T.append(t0)

# Create a DataFrame of forecast errors
edf = pd.DataFrame(e)

# Compute RMSFE (Root Mean Squared Forecast Error)
rmsfe = np.sqrt(edf.apply(np.square).mean())
print("\n============================")
print("      RMSFE EVALUATION      ")
print("============================")
print(rmsfe)
print("============================")


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

# First subplot: Comparison Between Actual Data and Forecasts
axes[0].plot(T, np.array(Y_actuals).mean(axis=1), marker='o', label="Actual Data", color="blue")
axes[0].plot(T, np.array(Y_forecasts).mean(axis=1), marker='o', linestyle="dashed", label="Forecasts", color="red")
axes[0].set_xlabel("Date", fontsize=12)
axes[0].set_ylabel("Target Value", fontsize=12)
axes[0].set_title("Comparison Between Actual Data and Forecasts", fontsize=14)
axes[0].legend()
axes[0].grid(True)

# Second subplot: Forecast Error Evolution Over Time
axes[1].plot(T, edf.mean(axis=1), marker='o', label="Average Forecast Error", color="green")
axes[1].set_ylabel("Forecast Error", fontsize=12)
axes[1].set_title("Forecast Error Evolution Over Time", fontsize=14)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()




# We import pandas library to better handle and manage the data tables
import pandas as pd # For managing data in tables (DataFrame)
import numpy as np # For advanced math calculations
import matplotlib.pyplot as plt  # For data visualization
import matplotlib.dates as mdates  # For managing dates in plots

# ------------------------------- #
#1. DOWNLOADING THE FRED-MD DATASET 
# ------------------------------- #

file_path = "/Users/isabellabonetto/Desktop/macroeconometrics_lab1.1/current.csv"
# Then we load dataset from local file
df = pd.read_csv(file_path)

# ------------------------------- #
#2.DATAFRAME CLEANING 
# ------------------------------- #
#The first row contains transformation codes of variables, so we remove it
df_cleaned = df.drop(index=0)

# We need to properly reset the index after dropping the first row
df_cleaned.reset_index(drop=True, inplace=True)

# We now convert the 'sasdate' column to datetime format for easier handling
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')

# We verify that the first few rows present data correctly loaded
print(df_cleaned.head()) 

# Dataset imported successfully✅: we can make additional checks in order to verify the dataset is correctly imported and transformed
print(df_cleaned.shape)  # It verifies the total number of rows and columns and we get 792 rows and 127 columns.
print(df_cleaned.columns) # We control column names
print(df_cleaned.dtypes) # We control the right format: datetime for sasdate and float64 for numerical data.
print(f"Total number of rows: {df_cleaned.shape[0]}")


# ------------------------------- #
# 3. TRANSFORMATION OF VARIABLES #
# ------------------------------- #

# Let's extract the transformation codes from the first line
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# df.iloc[0, 1:] → Selects the first row (index 0), excluding the first column (sasdate). This row contains the transformation codes.
# .to_frame() → Converts the extracted series into a DataFrame.
# .reset_index() → Turns the column names into a separate column called Series.
# transformation_codes.columns = ['Series', 'Transformation_Code'] → Renames the columns for better clarity.

#Now we have a DataFrame called transformation_codes with two columns:

#Series → The names of the economic variables.
#Transformation_Code → The transformation code that needs to be applied to each variable.

#To display this new dataframe we use the comand 
print(transformation_codes) #As we can see in this new dataframe we have 126 rows (because sasdate is exluded) and 2 columns

#Now let's analyse what are the transformnation codes

# Code 1: No transformation → The variable is already stationary, so no changes are needed.

# This means that the variable does not have a trend or changing variance over time.
# In this case, we do not need to transform the data.
#💡Example: Short-term interest rates, unemployment rates (often stationary in raw form).

# Code 2: First difference (ΔX_t = X_t - X_{t-1}) → Used for non-stationary series with a unit root, removing trends.

#Used for variables with a linear trend (random walk process).
#If a time series is non-stationary but becomes stationary after subtracting the previous value, we apply a first difference transformation.
#This removes linear trends and makes the series more stable.
#💡Example: GDP, stock prices, money supply.

# Code 3: Second difference (Δ²X_t = ΔX_t - ΔX_{t-1}) → Removes quadratic trends from the data.

#Used when first differencing is not enough (stronger trends).
#Some series have a quadratic trend (e.g., GDP growth), meaning that even after first differencing, they are still non-stationary.
#In this case, we apply a second difference.
#💡Example: GDP level, industrial production index.

# Code 4: Log transformation (log(X_t)) → Stabilizes variance, often used for financial or economic data with exponential growth.

#Used to stabilize variance (heteroskedasticity).
#If a time series has an increasing variance over time (e.g., stock prices, inflation), taking the log makes it more stable.
#This transformation is common in finance and macroeconomics.
#💡Example: Stock market indices, price indices (CPI, PPI), real GDP.

# Code 5: First difference of log (Δlog(X_t) = log(X_t) - log(X_{t-1})) → Equivalent to the percentage change of the variable.

#Used when a variable follows an exponential growth pattern.
#The first difference of a log-transformed series approximates the percentage change (growth rate).
#This is especially useful for variables like GDP, stock prices, and inflation rates.
#💡Example: GDP growth, inflation rate, productivity growth.

# Code 6: Second difference of log (Δ²log(X_t)) → Used for series with strong exponential trends.

#Used when the first difference of log is not enough to make a series stationary.
#If Δlog(X_t) is still non-stationary, we difference it again.
#This removes nonlinear trends and stabilizes highly volatile variables.
#💡Example: Inflation acceleration, productivity growth rate.

# Code 7: Approximate percentage change (Δ((X_t / X_{t-1}) - 1)) → A refined way to measure growth rates in certain economic indicators.

#Used as an alternative to the first difference of log.
#Instead of using Δlog(X_t), this transformation calculates the relative change in a variable.
#It is useful when we want an approximation of percentage changes but without logarithms.
#💡Example: Inflation rate, exchange rate fluctuations.


#Function to apply transformations based on the transformation code
#This function takes a time series (series) and a transformation code (code).
#It applies the appropriate transformation based on the code (from 1 to 7).
#If an invalid code is provided, it raises an error.



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

#Applying the transformations to each column in df_cleaned based on transformation_codes, this function has diffent purposes
#1️⃣ Iterates over each series (column name) and its corresponding transformation code from the transformation_codes DataFrame.
#The loop for series_name, code in transformation_codes.values: goes through all economic variables and their assigned transformation codes.
#2️⃣ Converts the column values to float to avoid type errors.
#Some values might be read as strings, so astype(float) ensures the data is numeric before applying transformations.
#3️⃣ Applies the apply_transformation function to transform the data based on the specified code.
#If a variable has code 2, the first difference is computed.
#If it has code 5, the log difference is applied.
#The function ensures that each variable is transformed correctly.

for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

#We now remove the first two rows 
#Many of the transformations introduce missing values (NaN) at the beginning of the time series.
#💡Examples:
#First difference (ΔX_t = X_t - X_{t-1})
#The first observation will be missing because there's no previous value to subtract.
#Second difference (Δ²X_t = ΔX_t - ΔX_{t-1})
#The first two observations will be missing.
#Log transformation + differences (log(X_t) - log(X_{t-1}))
#Also introduces NaN values in the first row(s).
#By removing the first two rows, we eliminate these missing values and ensure that the dataset starts with valid data.
df_cleaned = df_cleaned[2:]  

#After removing rows, the index will still reference the original row numbers (e.g., it might start from 2 instead of 0).
#reset_index(drop=True, inplace=True) does two things:
#drop=True → Prevents the old index from being added as a new column.
#inplace=True → Modifies the DataFrame directly instead of creating a copy.
df_cleaned.reset_index(drop=True, inplace=True)

#To check the transformed data 
df_cleaned.head() #As we can see by running this comand the 0th observastion starts from March 1959 and not anymore from January 1959
#with the above comand we had modified the file df_cleaned and we did not create a different copy


# ------------------------------- #
# 4. DATA VISUALIZATION #
# ------------------------------- #
#Now we want to create some graphs of some variables of our dataframe


#Let's consider three economic series to visualize (INDPRO, CPIAUCSL, TB3MS) and assign them human-readable names (“Industrial Production”, “Inflation (CPI)”, “3-month Treasury Bill rate.”)
series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS'] 
series_names = ['Industrial Production', 'Inflation (CPI)', '3-month Treasury Bill rate'] 
# Note that the order of the lists matches so that each name corresponds to its variable.

#Creating a plot with multiple subplots (one for each selected series)
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))
# plt.subplots(rows, columns, figsize=(width, height)) → Creates a figure with multiple subplots.
# len(series_to_plot) → Number of rows (since we have 3 time series, we create 3 subplots).
# 1 → We create a single-column layout (subplots stacked vertically).
# figsize=(8, 15) → Sets the size of the entire figure (8 inches wide, 15 inches tall).

# We check if axs is a list
axs = np.atleast_1d(axs)
# We control the list 
print(len(series_to_plot), len(series_names))  #They should have the same lenght

#We Run this long comand from point 4️⃣ to 1️⃣1️⃣
                     
                                                                             #4️⃣We check if the series exists in each series df_cleaned DataFrame columns.  
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):   # zip(axs, series_to_plot, series_names) → Loops over the subplots (ax), dataset columns (series_name), and their human-friendly names (plot_title).                                                                           
    if series_name in df_cleaned.columns:                                    # #if series_name in df_cleaned.columns: → Ensures we only plot series that exist in the dataset (avoids errors).
        
                                                                             #5️⃣We convert the sasdate column to datetime format (not necessary, since sasdate was converter earlier). 
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')     #This comand ensures that the sasdate column is recognized as a date (useful for proper plotting).
        
                                                                             #6️⃣We plot each series against the sasdate on the corresponding subplot, labeling the plot with its human-readable name. 
        ax.plot(dates, df_cleaned[series_name], label=plot_title)            #The general comand is ax.plot(x_values, y_values, label='Legend Name') Where: 
                                                                             # x_values = dates (time on the x-axis).
                                                                             # y_values = df_cleaned[series_name] (economic indicator on the y-axis).
                                                                             # label=plot_title → Uses the human-readable name for the legend.
        
                                                                             #7️⃣We now format the x-axis to display ticks and label the x-axis with dates taken every five years. 
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))               #This comand places major ticks (x-axis labels) every 5 years.    
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))             #This comand formats the x-axis ticks to show only the year (YYYY)

                                                                             #8️⃣Each subplot is titled with the name of the economic indicator.
        ax.set_title(plot_title)                                             #This comand adds a title to the subplot using the human-readable name.
        
                                                                             #9️⃣We label the x-axis “Year,” and the y-axis “Transformed Value,” to indicate that the data was transformed before plotting.
        ax.set_xlabel('Year')                                                #This comand labels the x-axis as "Year."
        ax.set_ylabel('Transformed Value')                                   #This comand labels the y-axis to indicate that the values were transformed before plotting.

                                                                             #🔟A legend is added to the upper left of each subplot for clarity.
        ax.legend(loc='upper left')                                          #This comand places the legend in the top-left corner of each subplot.

                                                                             #1️⃣1️⃣We rotate the x-axis labels by 45 degrees to prevent overlap and improve legibility.
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')    #This comand rotates the year labels 45 degrees to prevent overlap. (ha='right') aligns the labels to the right for better readability.

    else:
        ax.set_visible(False)                                                #This final comand hides plots for which the data is not availabl



#1️⃣2️⃣We use plt.tight_layout() automatically adjusts subplot parameters to give specified padding and avoid overlap.
plt.tight_layout()

#1️⃣3️⃣ We use plt.show() that displays the final figure, whose size is 640x480 pixels
plt.show()

# ------------------------------- #
# 5. ARX MODEL CONSTRUCTION #
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
# 6. MODEL ESTIMATION AND FORECASTING #
# ------------------------------- #


from numpy.linalg import solve # This line imports the function called solve from the NumPy library’s linear algebra module: in our model, we need to solve an equation to get the best-fitting numbers (coefficients) for our regression. This is a key part of estimating our model using Ordinary Least Squares (OLS)

# Estimating model parameters using the Ordinary Least Squares (OLS) method: 
beta_ols = solve(X.T @ X, X.T @ y) # This line calculates the coefficients of our forecasting model solving for the OLS estimator beta: (X'X)^{-1} X'Y; where X.T means the transpose of matrix X (flipping rows and columns), @ is the operator for matrix multiplication and solve(A, B) finds the solution β in the equation Aβ = B

# Making a one-step-ahead prediction
forecast = X_T @ beta_ols * 100 # X_T @ beta_ols multiplies the most recent row of data (X_T) by our coefficients (beta_ols) to get a forecasted value; * 100 gives the forecast in percentage points → we predict the next period's industrial production using the current available information
print(f"One-step ahead forecast: {forecast[0]:.2f}%") # Shows the forecasted value on the screen
forecast # This variable contains now the one-step ahead (h = 1 forecast in this case) of INDPRO expressed in percentage change. Showing this forecast allows us to later compare our predicted value with the actual value, evaluate the model's performance, and ultimately make informed economic decisions.
# Essentially we predicted what industrial production would be for the next time period, and our model’s guess came out as approximately 0.39%. This value is then used to calculate errors and evaluate the model's overall forecasting performance over multiple periods.

# ------------------------------- #
# 7. MODEL EVALUATION #
# ------------------------------- #
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
    ## Calculate the RMSFE, that is, the square root of the MSFE
np.sqrt(edf.apply(np.square).mean()) # edf.apply(np.square) squares each error in the DataFrame; .mean() calculates the average of these squared errors, giving the Mean Squared Forecast Error (MSFE); np.sqrt(...) takes the square root of the MSFE to produce the RMSFE, that is a common metric used to assess forecast accuracy. A lower RMSFE indicates that the forecasts are generally close to the actual values, meaning the model is performing well.

# ------------------------------- #
# 8. FORECAST VISUALIZATION #
# ------------------------------- #









# - - - - MATTIA's PART
num_predictions = 12
predictions = []
for _ in range(num_predictions):
    forecast = X_T @ beta_ols
    predictions.append(forecast[0])
    X_T = np.roll(X_T, shift=-1)
    X_T[-1] = forecast[0]

print(f"📈 Forecasts for the next {num_predictions} months:")
print(predictions)

# ------------------------------- #
# 7. MODEL EVALUATION #
# ------------------------------- #

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Computing prediction errors
y_pred = X @ beta_ols
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# ------------------------------- #
# 8. FORECAST VISUALIZATION #
# ------------------------------- #

plt.figure(figsize=(12,6))
plt.plot(df_cleaned['sasdate'].iloc[num_lags:-num_leads], y, label="Dati Reali", color="blue")
plt.plot(df_cleaned['sasdate'].iloc[num_lags:-num_leads], y_pred, label="Previsioni", color="red", linestyle="dashed")

plt.xlabel("Data", fontsize=12)
plt.ylabel("Produzione Industriale", fontsize=12)
plt.title("Confronto tra Dati Reali e Previsioni", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy.linalg import solve

# Selecting and extracting data from 2020 to 2025
df_filtered = df_cleaned[(df_cleaned['sasdate'] >= '2020-01-01') & (df_cleaned['sasdate'] <= '2025-12-31')]

# Choosing the target and predictive variables
y = df_filtered['INDPRO']
y_pred = y.shift(-1)  # Forecast simulation with ARX model

# Calculating errors
mse = mean_squared_error(y[:-1], y_pred.dropna())
rmse = np.sqrt(mse)
mae = mean_absolute_error(y[:-1], y_pred.dropna())

# Improved error display
print("============================")
print("      ERROR EVALUATION    ")
print("============================")
print(f"MSE  (Mean Squared Error)  : {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE  (Mean Absolute Error) : {mae:.4f}")
print("============================")

# ------------------------------- #
# 9. FILTERED CHART VISUALIZATION
# ------------------------------- #
plt.figure(figsize=(12,6))
plt.plot(df_filtered['sasdate'], y, label="Actual Data", color="blue")
plt.plot(df_filtered['sasdate'], y_pred, label="Forecast", color="red", linestyle="dashed")

plt.xlabel("Date", fontsize=12)
plt.ylabel("Industrial Production", fontsize=12)
plt.title("Comparison between Actual Data and Forecast (2020-2025)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
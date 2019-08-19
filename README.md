
# Time Series Modeling - Example Project

This project uses everything we've learned about Time Series Analysis to forecast the median home sales price by zipcode across the United States. Comments have been added to the code to explain what is happening at each step. 

### (Private--Do not release to students)

## Step 1: Import Necessary Packages

Needed for this lab:

* pandas, numpy, matplotlib for the normal stuff
* ARIMA model from statsmodels
* Helper functions from statsmodels, for plotting Time Series (optional)
* TQDM, for visualizing progress bars for long runtimes (optional)


```python
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import operator
%matplotlib inline
```

# Step 2: Load and Inspect Data

Load data directly from `zillow_data.csv`. Cast `'RegionName'` column to `str` during loading process. 


## A Note on Null Values

This dataset contains missing values. There are three possible cases for null values in this dataset:

1. A sequence of missing values at the start of the time series. This is seen with newer zipcodes, which will have missing values for any date before the creation of that zipcode. 
2. A sequence of missing values at the end of the time series. This is seen with zipcodes that were dissolved after a rezoning. 
3. Spontaneous missing values, with no pattern. These are just run-of-the-mill missing values.

Only null values of type '3' can be imputed safely. Please note that this is not done in this example notebook--instead, all zipcodes with any missing values are simply dropped. 


```python
# Import and inspect the data
raw_data_df = pd.read_csv('zillow_data.csv', dtype={'RegionName': 'str'})
raw_data_df.head()
```

# Step 3: Drop Unneeded Columns

We don't need `'SizeRank'` or `'RegionID'`, so let's drop them. 


```python
df1 = raw_data_df.drop(['SizeRank', 'RegionID'], axis=1, inplace=False)
df1.head()
```

# Example: Filtering Zipcodes for Arkansas

Zipcodes were retrieved by searching city zipcodes on Google. Store each as a string inside separate lists. 


```python
## Zipcodes for AK metro areas
hot_springs = ['71901', '71902', '71903', '71913', '71914']
little_rock = ['72002', '72203', '72207', '72212', '72219', '72225', '72260', '72103', 
               '72204', '72209', '72214', '72221', '72227', '72295', '72201', '72205', 
               '72210', '72215', '72222', '72231', '72202', '72206', '72211', '72217', 
               '72223', '72255']
fayetteville = ['72701', '72702', '72703', '72704', '72730', '72764']
searcy = ['72082', '72143', '72145', '72149']

ar_city_zipcodes = hot_springs + little_rock + fayetteville + searcy

ar_city_zipcodes
```


```python
ar_cities_df = df1[df1['RegionName'].isin(ar_city_zipcodes)]
ar_cities_df
```

# Example: Filtering by City Name

Can also get zipcodes for a given city by filtering by the name of the city. 


```python
# Check that zipcodes starting with 0 are now formatted correctly
df1[df1['City'] == 'Agawam']
```

# Step 4: Creating Time Series Plots

## Step 4.1: Filter by Zipcodes

In this example, we slice the relevant zipcodes for each of our example cities into separate DataFrames. This will make visualizing them easier. 


```python
# Create data sets containing zip codes for AR metro area
searcy_df = df1[df1['RegionName'].isin(searcy)]
littlerock_df = df1[df1['RegionName'].isin(little_rock)]
fayetteville_df = df1[df1['RegionName'].isin(fayetteville)]
hotsprings_df = df1[df1['RegionName'].isin(hot_springs)]

display(searcy_df)
display(littlerock_df)
display(fayetteville_df)
display(hotsprings_df)
```

## Step 4.2: Remove Unneeded Columns

For our Time Series Plots, we only need the `RegionName` and the actual median housing values for each.  This means that we can drop everything else. 


```python
cols_to_drop = ['City', 'State', 'Metro', 'CountyName']
searcy_clean_df = searcy_df.drop(cols_to_drop, axis=1, inplace=False)
fayetteville_clean_df = fayetteville_df.drop(cols_to_drop, axis=1, inplace=False)
hotsprings_clean_df = hotsprings_df.drop(cols_to_drop, axis=1, inplace=False)
littlerock_clean_df = littlerock_df.drop(cols_to_drop, axis=1, inplace=False)
```

# Step 4.3: Get Datetimes

We will use the datetimes as our indices for these plots. Currently, the column names for each remaining column that is not the zipcode in question is contains the datetimes. These datetimes are stored as strings, in the format `'%Y-%m'`. 

In the cell below, we create a series called `datetimes` using pandas and the column values for one of the cities (we exclude the first column value, since this contains `'RegionName'`).


```python
datetimes = pd.to_datetime(searcy_clean_df.columns.values[1:], format='%Y-%m')
```

# Step 4.4: Set Params for Matplotlib Visualizations

This step is optional, but will make our plots easier to read.


```python
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
```

# Step 4.5: Create a Function for Visualizing Time Series Data

This one is a bit complicated, so the code is commented and broken down line by line. 

**_NOTE_**: For the example below, we've only visualized the data for 1997 through 2013. This was done on purpose, to show how to visualize select ranges of data. The index values were obtained by manually looking at the date ranges and determining that `[10:215]` includes everything between 1997 and 2013. 


```python

def time_series(df, name=None, legend=None):
    
    # Instantiate a figure object. 
    plt.figure()
    if not legend:
        legend = list(df['RegionName'])
    # Enumerate through each row in the dataframe passed in. Each row is a different zipcode.
    for ind, row in df.iterrows():
        
        # Get the median housing value data for the date ranges we want and store in a Series object
        data = pd.Series(row.iloc[10:215])
        # Set the appropriate datetimes for data as the index for our data series
        data.index = datetimes[10:215]
        # Plot data for current zipcode on figure we instantiated on line 4. Set xticks to corresponding datetimes
        # Also make the figure large, so that we can read it more easily
        ax = data.plot(figsize=(20, 10), xticks=datetimes[10:215])
        # add a label
        plt.ylabel("Median Sales Value ($)")
        # let matplootlib autoformat the datetimes for the xticks
        plt.gcf().autofmt_xdate()
        
        # If name of city was provided, use it to set title of plot
        if name:
            plt.title("Median Home Value by Zip Code in {} from 1997-2013".format(name))
        else:
            plt.title("Avg Median Home Value in AR Metro Area, 1997-2013")
        
    plt.legend(legend)
            
    plt.show()
        

time_series(fayetteville_clean_df, name='Fayetteville', legend=fayetteville)
# time_series(searcy_clean_df, name='Searcy')
# time_series(hotsprings_clean_df, hot_springs, 'Hot Springs')
# time_series(littlerock_clean_df, little_rock, 'Little Rock')
```

# Step 4.6: Visualizing the Average Median Home Sale Price for a Collection of Zipcodes

To visualize the average median home sales value for an area, we can use the function we created above, but we need to do a bit of processing first to get it into the shape needed. 

1. First, we concatenate all of the dataframes containing the zipcodes we want to average.
2. Next, we create a new DataFrame containing a single column of data called `'Avg_Median_Value'` for the date range we want (in this example, still focusing only on values between 1997-2013). 
3. Next, drop the `'RegionName'` column.
4. Finally, inspect the data to see what our newly computed `'Avg_Median_Value'` data looks like. 


```python
arkansas_metro_df = pd.concat([searcy_clean_df, littlerock_clean_df, fayetteville_clean_df, hotsprings_clean_df])
avg_metro_value_df = pd.DataFrame(arkansas_metro_df[10:215].mean(), columns=['Avg_Median_Value'])
avg_metro_value_df.drop('RegionName', axis=0, inplace=True)
avg_metro_value_df.head()
```

The data looks fine, but it need to be transposed in order to work with the function we've written. 

Note that we can chance the value of our legend to whatever string we want by wrapping it in an array and passing it in to the `legend` parameter.


```python
time_series(avg_metro_value_df.transpose(), name="Average Median Value", legend=['Avg Median Sale Value'])
```

# ARIMA Modeling 

The next section demonstrates how to do ARIMA modeling on this data set. 

## 'Melting' the Data

In order to train the model, we need to first **_melt_** the data into the appropriate shape. ARIMA models expect the data in columnar format ("long"), and in our current format, the values are stored in rows ("wide"). 

The cell below shows some sample code for melting a dataframe, and displays the same dataframe in both wide (unmelted) and long (melted) formats. 


```python
melted = pd.melt(searcy_df, id_vars=['RegionName', 'City', 'State', 'Metro', 'CountyName'], var_name='time')
melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
melted = melted.dropna(subset=['value'])

display(searcy_df.head())
melted.head(10)
```

# Step 5: Create a Function for Melting Data

Since this is an operation we'll need to for any group of data we want to format for ARIMA modeling, we should create a function in order to save ourselves some time. 


```python
def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionName', 'City', 'State', 'Metro', 'CountyName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})
```

# Step 6: Creating a Function to Evaluate Results

Before we actually fit the model, we'll create a function that creates predictions for datetimes with known values based on the previous data, and then compare the lagged predictions with ground truth values from our time series data. 


```python
def get_results(df, preds, name):
    if 'pandas.core.frame.DataFrame' in str(type(df)):
        current_price = df.iloc[-1].value
    else:
        current_price = df[-1]
    year_later = preds[11]
    year_3_val = preds[35]
    year_5_val = preds[-1]

    print("Current Avg Median Home Value in {}: ${:.2f}".format(name, current_price))
    print("Predicted Avg Median Home Value for {} in April 2019: ${:.2f}".format(name, year_later))
    expected_appreciation_value_1 = year_later - current_price
    expected_appreciation_percent_1 = expected_appreciation_value_1 / current_price
    expected_appreciation_value_3 = year_3_val - current_price
    expected_appreciation_percent_3 = expected_appreciation_value_3 / current_price
    expected_appreciation_value_5 = year_5_val - current_price
    expected_appreciation_percent_5 = expected_appreciation_value_5 / current_price

    print("Expected property value appreciation for 1 year in {} :  ${:.2f}".format(name, expected_appreciation_value_1))
    print("Expected Return on Investment after 1 year:  {:.4f}%".format(expected_appreciation_percent_1 * 100))
    print("Expected property value appreciation for 3 years in {} :  ${:.2f}".format(name, expected_appreciation_value_3))
    print("Expected Return on Investment after 3 years:  {:.4f}%".format(expected_appreciation_percent_3 * 100))
    print("Expected property value appreciation for 5 years in {} :  ${:.2f}".format(name, expected_appreciation_value_5))
    print("Expected Return on Investment after 5 years:  {:.4f}%".format(expected_appreciation_percent_5 * 100))
```

# Step 7: Fitting Our ARIMA Model

Finally, we create a `fit_model()` function that takes in our (melted!) dataframe, the zipcode (for display purposes), and an optional parameter for visualizing the results of our model's fit. 

The function below has been commented to explain what is happening at each step. 


```python
def fit_model(df, zipcode, show_graph=True):
    # Get only the values from the dataframe
    vals = df.values
    # Split the data into training and testing sets by holding out dates past a certain point. Below, we use index 261 for 
    # this split
    train = vals[:261]
    test = vals[261:]
    
    # Use a list comprehension to create a "history" list using our training data values
    history = [i for i in train]
   
    # initialize an empty list for predictions
    preds = []
    
    # loop through a list the length of our training set
    for i in range(len(test)):
        
        # create an ARIMA model and pass in our history list. Also set `order=(0,1,1)` (order refers to AR and MA params--
        # see statsmodels documentation for ARIMA for more details)
        model = ARIMA(history, order=(0,1,1))
        
        # Fit the model we just created
        fitted_model = model.fit(disp=0)
        # Get the forecast of the next value from our fitted model, and grab the first value to use as our 'y-hat' prediction
        output = fitted_model.forecast()
        y_hat = output[0]
        
        # append y_hat to our list of predictions
        preds.append(y_hat)
        obs = test[i]
        
        # Get the actual ground truth value for this datetime and append it to the history array
        history.append(obs)
    
    
    # get the forecast for the next three years (1 month==1 timestep in our data)
    future_preds = fitted_model.forecast(steps=36)[0]

    # Visualize the ARIMA model's predictions vs the actual ground truth values for our test set
    if show_graph == True:
        print('Predicted: {} \t Expected: {}'.format(y_hat, obs))
        # Also calculate the MSE
        mse = mean_squared_error(test, preds)
        print("MSE for Test Set: {}".format(mse))
        plt.plot(test)
        plt.plot(preds, color='r')
        plt.ylabel('Median Home Value ($)')
        plt.title('Predicted vs Expected Median Home Sale Values'.format(zipcode))
        plt.legend(['Actual', 'Predicted'])
        plt.show()

        
        plt.figure()
        plt.plot(future_preds)
        plt.ylabel('Median Home Value ($)')
        plt.title('Predicted Home Value, {}, Next 36 Months'.format(zipcode))
        plt.show()
        get_results(df, future_preds, zipcode)
        
    return future_preds
```


```python
aggregate_df = melt_data(df1)
aggregate_df.head()
```


```python
_ = fit_model(aggregate_df, "US")
```

# Optional: Compare Forecasts for Every Zipcode in US

The following cells demonstrate how to use all the code written so far to create and compare 5-year forecasts for every zipcode in the dataset. Note that this is well outside the scope of the project!

**_NOTE: Running the cells below takes >1 hour on a fast computer!_**


```python
def model_data_by_zip(df, num_top_zips=3):
    
    df.dropna(axis=0, inplace=True)
    zip_roi_12_month = {}
    zip_roi_36_month = {}
    zip_roi_60_month = {}
    
    # Get 12-month RoI for each zipcode
    with tqdm(total=len(list(df.iterrows()))) as pbar:
        for ind, row in df.iterrows():
            pbar.update(1)
            series = pd.Series(row)
            name = series[0]
            data = series[5:]

            preds_for_zip = fit_model(data, name, show_graph=False)
            last_val = row[-1]
            predicted_val_12 = preds_for_zip[11]
            predicted_val_36 = preds_for_zip[35]
            predicted_val_60 = preds_for_zip[-1]
            roi_12 = (predicted_val_12 - last_val) / last_val
            roi_36 = (predicted_val_36 - last_val) / last_val
            roi_60 = (predicted_val_60 - last_val) / last_val
            zip_roi_12_month[name] = roi_12
            zip_roi_36_month[name] = roi_36
            zip_roi_60_month[name] = roi_60
    
    # Sort dict by values and return amount specified by optional parameter, default 3
    sorted_by_roi_12 = sorted(zip_roi_12_month.items(), key=operator.itemgetter(1), reverse=True)
    sorted_by_roi_36 = sorted(zip_roi_36_month.items(), key=operator.itemgetter(1), reverse=True)
    sorted_by_roi_60 = sorted(zip_roi_60_month.items(), key=operator.itemgetter(1), reverse=True)
    
    return (sorted_by_roi_12[:num_top_zips], sorted_by_roi_36[:num_top_zips], sorted_by_roi_60[:num_top_zips])
```


```python
def format_results(results):
    results_12 = results[0]
    results_36 = results[1]
    results_60 = results[2]
    
    print("Top Zip Codes for Predicted RoI--1 Year")
    
    for zipcode, roi in results_12:
        print("Zipcode: {} \t Predicted 12-month RoI: {:.6f}%".format(zipcode, roi * 100))
    
    print("")
    print('-' * 60)
    print("")
    
    print("Top Zip Codes for Predicted RoI--3 Years")
    
    for zipcode, roi in results_36:
        print("Zipcode: {} \t Predicted 36-month RoI: {:.6f}%".format(zipcode, roi * 100))
        
    print("")
    print('-' * 60)
    print("")
    
    print("Top Zip Codes for Predicted RoI--5 Years")
    
    for zipcode, roi in results_60:
        print("Zipcode: {} \t Predicted 60-month RoI: {:.6f}%".format(zipcode, roi * 100))
        
```


```python
# run model on every zipcode 
# (model drops rows containing any null values)

top_zips_in_us = model_data_by_zip(df1, num_top_zips=10)
```


```python
format_results(top_zips_in_us)
```

import os
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import xarray as xr
from multiprocessing import Pool
from tqdm import tqdm
from joblib import Parallel, delayed

# Configuration
time_interval_reconstruction = ["2018-01-01 00:00:00", "2019-12-31 23:54:00"]
predict_variable = 'WSPDchyv2_2011_2019'
k = 3
folder_path = '/Users/Murilo/weather_data'
netCDF_resolution = 1

# Start
all_dataframes = {}  # This dictionary will store DataFrames for each nc file

nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]

for nc_file in nc_files:
    with xr.open_dataset(os.path.join(folder_path, nc_file)) as ds:
        data_dict = {}
        
        # Dynamically load all variables from the file
        for var_name, variable in ds.data_vars.items():
            data = variable.values
            
            data = np.where(data == 9.969210e+36, np.nan, data)
            fraction_available = np.sum(~np.isnan(data)) / data.size

            # If more than 80% of the data is available, store it in the dictionary
            if fraction_available > 0.80:
                data_dict[var_name] = data

        # Convert the time variable
        if 'time' in ds.coords:
            time_data = ds['time']   
            
            # Add time data to the dictionary
            data_dict['time'] = time_data
        
        # Convert the data dictionary to a DataFrame
        df = pd.DataFrame(data_dict)
        
        # Store this DataFrame in our all_dataframes dictionary
        all_dataframes[nc_file] = df

all_dataframes

merged_df = None

for filename, df in all_dataframes.items():
    # Skip the 'time' column for renaming
    rename_dict = {col: col + filename[:-3] if col != 'time' else col for col in df.columns}
    df = df.rename(columns=rename_dict)

    if merged_df is None:
        merged_df = df
    else:
        # Merge the DataFrame with the main merged_df on 'time'
        merged_df = pd.merge(merged_df, df, on='time', how='outer')

print(merged_df.head())

start_time = merged_df['time'].iloc[0]
end_time = merged_df['time'].iloc[-1]

print(f"Start Time: {start_time}")
print(f"End Time: {end_time}")

# Extract client times
client_start_time = pd.Timestamp(time_interval_reconstruction[0])
client_end_time = pd.Timestamp(time_interval_reconstruction[1])

# Check if client times fall within the dataframe's time range
if client_start_time < start_time or client_end_time > end_time:
    print("⚠️ WARNING ⚠️")
    print("The time interval provided is OUTSIDE the range of the available data.")
    print(f"Available data range: {start_time} to {end_time}")
    print(f"Interval: {client_start_time} to {client_end_time}")
    print("Please provide a valid time range.")
else:
    print("✅ SUCCESS ✅")
    print("The time interval provided is INSIDE the range of the available data.")

# Splitting into training and testing based on the client's time range
training_data = merged_df[(merged_df['time'] < client_start_time) | (merged_df['time'] > client_end_time)]
testing_data = merged_df[(merged_df['time'] >= client_start_time) & (merged_df['time'] <= client_end_time)]

# Make a copy of the original dataframes for later use
training_with_na = training_data.copy()
testing_with_na = testing_data.copy()

# Drop NaNs for training and testing datasets
training_na = training_data.dropna()
testing_na = testing_data.dropna()

# Define X_train and Y_train
X_train = training_na.drop(columns=['time', predict_variable])
Y_train = training_na[predict_variable]

# Define X_val and Y_val
X_val = testing_na.drop(columns=['time', predict_variable])
Y_val = testing_na[predict_variable]
merged_df.to_csv('merged_df.csv', index=False)

# 1. Combine training_with_na and testing_with_na to get full dataset
full_data_with_na = pd.concat([training_with_na, testing_with_na], axis=0)

# Define the PLS model
pls = PLSRegression(n_components=2)
pls.fit(X_train, Y_train)

# Fit the PLS model on training data
X_all = pd.concat([X_train, X_val], axis=0)

# 2. Transform the data
X_all_transformed = pls.transform(X_all)

# Convert transformed data to DataFrame for easier manipulation
X_all_transformed_df = pd.DataFrame(X_all_transformed, index=X_all.index)

# Merge with the original timestamps
result = pd.merge(full_data_with_na[['time']], X_all_transformed_df, left_index=True, right_index=True, how='left')

# 1. Combine training_with_na and testing_with_na to get full dataset
full_data_with_na = pd.concat([training_with_na, testing_with_na], axis=0)

# Define the PLS model
pls = PLSRegression(n_components=2)
pls.fit(X_train, Y_train)

# Fit the PLS model on training data
X_all = pd.concat([X_train, X_val], axis=0)

# 2. Transform the data
X_all_transformed = pls.transform(X_all)

# Convert transformed data to DataFrame for easier manipulation
X_all_transformed_df = pd.DataFrame(X_all_transformed, index=X_all.index)

# Merge with the original timestamps
result = pd.merge(full_data_with_na[['time']], X_all_transformed_df, left_index=True, right_index=True, how='left')

assert merged_df.shape[0] == result.shape[0], "Dataframes have different sizes!"

# Use a coluna 'time' de 'merged_df' para criar um array booleano
time_conditions = (merged_df['time'] < client_start_time) | (merged_df['time'] > client_end_time)

Y_analogues = Y_all_df[time_conditions]
Y_pred = Y_all_df[~time_conditions]

# Convert dataframes to numpy arrays for faster computation
Y_analogues_np = Y_analogues.values
Y_pred_np = Y_pred.values

# Function to compute the most similar row for a single row in Y_pred
def find_most_similar(row):
    distances = np.nansum((Y_analogues_np - row) ** 2, axis=1)
    return np.argmin(distances)

# Use joblib to parallelize the operation
n_jobs = 8  # This will use all CPU cores. Adjust as needed.
similar_rows = Parallel(n_jobs=n_jobs)(delayed(find_most_similar)(row) for row in tqdm(Y_pred_np))

Y_pred['most_similar_row_in_Y_analogues'] = similar_rows

# End

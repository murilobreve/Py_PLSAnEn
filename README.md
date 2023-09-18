# Weather Data Analysis with Partial Least Squares Regression

This code provides a systematic approach to handling and analyzing large-scale weather datasets stored in the `.nc` format.

<img src="https://github.com/murilobreve/Py_PLSAnEn/assets/68515365/768c629e-10cf-4ac8-8a23-05c9dd1f0ad5" width="200">

## Table of Contents

- [Prerequisites](#prerequisites)
- [Execution Overview](#execution-overview)
- [Steps](#steps)
  - [Configuration](#configuration)
 
## Prerequisites

Ensure you have the following Python packages installed:

os
netCDF4
numpy
pandas
datetime
sklearn
xarray
multiprocessing
tqdm
joblib

## Execution Overview

The script performs the following operations:

1. Loads data from `.nc` files located in a specified directory.
2. Processes and merges these datasets.
3. Validates the time range of data.
4. Splits the data into training and testing datasets.
5. Applies Partial Least Squares (PLS) Regression.
6. Transforms and validates the resultant data.
7. Finds the most similar data points using parallel computation.

## Steps

### Configuration

```python
time_interval_reconstruction = ["2018-01-01 00:00:00", "2019-12-31 23:54:00"]
predict_variable = 'WSPDchyv2_2011_2019'
k = 3
folder_path = '/Users/Murilo/weather_data'
netCDF_resolution = 1



# Import necessary library
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import joblib
import os
import yaml

def load_parameter(parameter_direction):
    with open(parameter_direction,'r') as file:
        params = yaml.safe_load(file)
    
    return params

# Load params
params = load_parameter("app/config/configuration_file_1.yaml")

# Show params
params

# Import dataset
dataset = pd.read_excel(params['data_direction'])

# Show dataset
dataset

# Sanity check
dataset.shape

# Save to pickle
joblib.dump(dataset, "app/processed/dataset.pkl")

# Check available column in dataframe
dataset.columns

# Check data type
dataset.info()

def non_numerical_finder(data, columns):
    
    """
    This function is to locate a non numerical value within dataset. 
    
    :param data: <pandas dataframe> data contain sample
    :param column: <string> column name
    :return non_numerical: <list> list contain indexes
    """
    
    # Create blank list
    non_numerical = []
    
    # Loop to find non numerical column
    for col in columns:
        non_numeric = pd.to_numeric(data[col], errors = 'coerce').isna()
        non_numerical.extend(non_numeric[non_numeric == True].index.tolist())
    
    # Remove duplicates
    non_numerical = list(set(non_numerical))
    
    return non_numerical

# Find non numerical index
non_numerical = non_numerical_finder(dataset, params['int32_columns'])

# Show table which row or index have non numerical value
dataset.loc[non_numerical]

# Replace value function
def replace_value(data, column, column_number, old_value, replace_value):
    
    """
    This function is to locate and replace value from a column. 
    
    :param data: <pandas dataframe> data contain sample
    :param column: <string> column name
    :param column_number: <string> single or multiple column
    :param old_value: <string/int/float> old value or current value
    :param replace_value: <string/int/float> value to replace old value
    :return data: <pandas dataframe> data contain sample
    """
    
    if str(column_number).lower() == "multiple":
        for col in column:
            data[col] = data[col].replace(old_value, replace_value)
    elif str(column_number).lower() == "single":
        data[column] = data[column].replace(old_value, replace_value)
        
    else: 
        "column_number is not correct"
        
    return data

# Handling non numeric data
replace_value(data = dataset, 
              column = params['int32_columns'],
              column_number = "multiple",
              old_value = "---", 
              replace_value = -1)

# Show table which row or index have non numerical value
dataset.loc[non_numerical]

# Find non numerical index
non_numerical_new = non_numerical_finder(dataset, params['int32_columns'])
non_numerical_new

# Creating function to replace value to a specific row and cell
def replace_specific_index(data, index, column, new_value):
    
    """
    This function is created to replace a new value to old value located in specific index and column
    :param data: <pandas dataframe> data contain sample
    :param index: <int> specific index
    :param column: <string> specific column
    :param new_value: <string/int/float> new value to replace old value
    :return dataset: <pandas dataframe> data contain sample
    """
    data.loc[index, column] = new_value
    
    return data

# Replace value
replace_specific_index(data = dataset, index = non_numerical_new[0], column = 'max', new_value = 49)
replace_specific_index(data = dataset, index = non_numerical_new[0], column = 'critical', new_value = 'PM10')
replace_specific_index(data = dataset, index = non_numerical_new[0], column = 'categori', new_value = 'BAIK')

# Handling non numeric for pm10, pm25, so2, co, o3, no2
def change_data_type(data, column, data_type):
    
    """
    This function is to change the type of a specific column into a desired type.
    
    :param data: <pandas dataframe> data contain sample
    :param column: <string> column name
    :param data_type: <string> desired data type
    :return dataset: <pandas dataframe> data contain sample
    """
    
    for col in column:
        data[col] = data[col].astype(data_type)
    
    return data

# Change column type
change_data_type(data = dataset, column = params['int32_columns'], data_type = int)

# Sanity check
dataset.info()

# Categori proportion
dataset[params['label']].value_counts()

# Handling irrelevant data
dataset.drop(index = dataset[dataset[params['label']] == "TIDAK ADA DATA"].index, inplace = True)

# Categori proportion
dataset[params['label']].value_counts()

# Sanity check
dataset.info()

# Save as pickle
joblib.dump(dataset, "app/processed/dataset_clean.pkl")

def check_data(input_data, params, api = False):

    if not api:
        # Check data types
        assert input_data.select_dtypes("datetime").columns.to_list() == \
            params["datetime_columns"], "an error occurs in datetime column(s)."
        assert input_data.select_dtypes("object").columns.to_list() == \
            params["object_columns"], "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            params["int32_columns"], "an error occurs in int32 column(s)."

    # check range of data
    assert set(input_data.stasiun).issubset(set(params["range_stasiun"])), "an error occurs in stasiun range."   
    assert input_data['pm10'].between(params["range_pm10"][0], params["range_pm10"][1]).sum() == len(input_data), "an error occurs in pm10 column."
    assert input_data['pm25'].between(params["range_pm25"][0], params["range_pm25"][1]).sum() == len(input_data), "an error occurs in pm25 column."
    assert input_data['so2'].between(params["range_so2"][0], params["range_so2"][1]).sum() == len(input_data), "an error occurs in so2 column."
    assert input_data['co'].between(params["range_co"][0], params["range_co"][1]).sum() == len(input_data), "an error occurs in co column."
    assert input_data['o3'].between(params["range_o3"][0], params["range_o3"][1]).sum() == len(input_data), "an error occurs in o3 column."
    assert input_data['no2'].between(params["range_no2"][0], params["range_no2"][1]).sum() == len(input_data), "an error occurs in no2 column."

# Check data
check_data(dataset, params, False)

# Input-Output split function
def input_output_split(df, column):
    
    """
    This function is created to split input and output column
    :param df: <pandas dataframe> data contain sample
    :param columns: <string> column name
    :return X: <pandas dataframe> data contain input
    :return y: <pandas dataframe> data contain output
    """
    
    X = df.drop(column, axis = 1)
    y = df[column]
    
    return X,y

# Input-output split
X, y = input_output_split(df = dataset, column = params['label'])

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.35,
                                                    stratify = y,
                                                    random_state = 123)

# Train-Test split
X_valid, X_test, y_valid, y_test = train_test_split(X_test,
                                                    y_test,
                                                    test_size = 0.5,
                                                    stratify = y_test,
                                                    random_state = 123)

# Save as pickle
joblib.dump(X_train, "app/processed/X_train.pkl")
joblib.dump(y_train, "app/processed/y_train.pkl")
joblib.dump(X_valid, "app/processed/X_valid.pkl")
joblib.dump(y_valid, "app/processed/y_valid.pkl")
joblib.dump(X_test, "app/processed/X_test.pkl")
joblib.dump(y_test, "app/processed/y_test.pkl")
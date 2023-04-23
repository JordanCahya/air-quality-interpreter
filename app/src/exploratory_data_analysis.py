# Import necessary library
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as scs
import joblib
import yaml
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings("ignore")

def load_parameter(parameter_direction):
    with open(parameter_direction,'r') as file:
        params = yaml.safe_load(file)
    
    return params

# Load params
params = load_parameter("app/config/configuration_file_1.yaml")

# Show params
params

# Load variables
X_train = joblib.load("app/processed/X_train.pkl")
y_train = joblib.load("app/processed/y_train.pkl")

# Store into variable
EDA_dataset_missing_value = pd.concat([X_train, y_train], axis = 1)

# Checking replaced value
for col in params['int32_columns']:
    print(len(EDA_dataset_missing_value[EDA_dataset_missing_value[col] == -1]))

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

# Replace value
EDA_dataset_missing_value = replace_value(data = EDA_dataset_missing_value, 
                                          column = params['int32_columns'],
                                          column_number = "multiple",
                                          old_value = -1, 
                                          replace_value = np.nan)

# Sanity check
EDA_dataset_missing_value.isna().sum()

# Store into variable
EDA_dataset_skewness = EDA_dataset_missing_value

# Check skewness
EDA_dataset_skewness.skew(numeric_only = True)

# Split dataset
EDA_dataset_baik = EDA_dataset_skewness[EDA_dataset_skewness[params['label']] == params['label_categories'][0]]
EDA_dataset_tidak_baik = EDA_dataset_skewness[EDA_dataset_skewness[params['label']] != params['label_categories'][0]]

# Descriptive analysis
EDA_dataset_baik.describe()

# Descriptive analysis
EDA_dataset_tidak_baik.describe()

# Checking data skewness
EDA_dataset_outliers = EDA_dataset_skewness

# Checking outliers using boxplot
sns.boxplot(data=EDA_dataset_outliers[params['int32_columns']])

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

# Store into variable
EDA_dataset_correlation = pd.concat([X_train, y_train], axis = 1)

# Split input and output
EDA_X_train, EDA_y_train = input_output_split(df = EDA_dataset_correlation, column = params['label'])

# Create Kendall's rank coefficient and p-value calculation function
def kendall_rank(input, output):
    
    """
    This function is created to calculate the level of significance between numeric type input data and object type output data 
    using kendall's rank
    
    :param input: <pandas dataframe> data contain input
    :param output: <pandas dataframe> data contain output  
    :return kendall_rank: <dictionary> collection of coefficient and p_value
    """
    
    # Create an empty dictionary to store the results
    kendall_rank = {}
    
    # Loop over each column in the numerical data DataFrame
    for col in input.columns:
        
        # Calculate Kendall's rank correlation coefficient    
        coefficient, p_value = kendalltau(input[col], output)
        
        # Store the results in the dictionary
        kendall_rank[col] = (coefficient, p_value)
    
    return kendall_rank

# Show Kendall's rank coefficient and p-value
kendall_rank(input = EDA_X_train[params['int32_columns']], output = EDA_y_train)

# Check data proportion using plot
sns.histplot(data = pd.concat([X_train, y_train], axis = 1), x = params['label'], hue = params['label'])
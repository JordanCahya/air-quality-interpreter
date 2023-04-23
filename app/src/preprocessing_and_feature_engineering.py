# Import necessary library

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as scs
import joblib
import yaml
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency

def load_parameter(parameter_direction):
    with open(parameter_direction,'r') as file:
        params = yaml.safe_load(file)
    
    return params

# Load params
params = load_parameter("app/config/configuration_file_1.yaml")

# Show params
params

# Load variables

# Input and output data train
X_train = joblib.load("app/processed/X_train.pkl")
y_train = joblib.load("app/processed/y_train.pkl")

# Input and output data valid
X_valid = joblib.load("app/processed/X_valid.pkl")
y_valid = joblib.load("app/processed/y_valid.pkl")

# Input and output data test
X_test = joblib.load("app/processed/X_test.pkl")
y_test = joblib.load("app/processed/y_test.pkl")

# Combining data
train_set = pd.concat([X_train, y_train], axis = 1)
validation_set = pd.concat([X_valid, y_valid], axis = 1)
testing_set = pd.concat([X_test, y_test], axis = 1)

# Check data proportion
train_set[params['label']].value_counts()

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
replace_value(train_set, params['label'], 'single', 'SEDANG', "TIDAK SEHAT")

# Check data proportion
train_set[params['label']].value_counts()

# Check data proportion
validation_set[params['label']].value_counts()

# Replace value
replace_value(validation_set, params['label'], 'single', 'SEDANG', "TIDAK SEHAT")

# Check data proportion
validation_set[params['label']].value_counts()

# Check data proportion
testing_set[params['label']].value_counts()

# Replace value
replace_value(testing_set, params['label'], 'single', 'SEDANG', "TIDAK SEHAT")

# Check data proportion
testing_set[params['label']].value_counts()

# Checking replaced value
for col in params['int32_columns']:
    print(len(train_set[train_set[col] == -1]))

# Replace value
replace_value(data = train_set, 
              column = train_set.columns,
              column_number = "multiple",
              old_value = -1,
              replace_value = np.nan)

# Sanity check
train_set.isna().sum()

# Checking replaced value
for col in params['int32_columns']:
    print(len(validation_set[validation_set[col] == -1]))

# Replace value
replace_value(data = validation_set, 
              column = validation_set.columns,
              column_number = "multiple",
              old_value = -1,
              replace_value = np.nan)

# Sanity check
validation_set.isna().sum()

# Checking replaced value
for col in params['int32_columns']:
    print(len(testing_set[testing_set[col] == -1]))

# Replace value
replace_value(data = testing_set, 
              column = testing_set.columns,
              column_number = "multiple",
              old_value = -1,
              replace_value = np.nan)

# Sanity check
testing_set.isna().sum()

# Calculate median
imputation_baik = int(train_set[train_set[params['label']] == params['label_categories'][0]].pm10.median())
imputation_tidak_sehat = int(train_set[train_set[params['label']] == params['label_categories'][1]].pm10.median())

# Show mean
imputation_baik, imputation_tidak_sehat

# Checking mising value proportion
train_set[train_set[params['label']] == params['label_categories'][0]].pm10.isnull().sum(), train_set[train_set[params['label']] == params['label_categories'][1]].pm10.isnull().sum()

# Missing value imputation
train_set.loc[train_set[(train_set[params['label']] == params['label_categories'][0]) & (train_set['pm10'].isnull() == True)].index, "pm10"] = imputation_baik
train_set.loc[train_set[(train_set[params['label']] == params['label_categories'][1]) & (train_set['pm10'].isnull() == True)].index, "pm10"] = imputation_tidak_sehat

# Checking mising value proportion
train_set[train_set[params['label']] == params['label_categories'][0]].pm10.isnull().sum(), train_set[train_set[params['label']] == params['label_categories'][1]].pm10.isnull().sum()

# Checking mising value proportion
validation_set[validation_set[params['label']] == params['label_categories'][0]].pm10.isnull().sum(), validation_set[validation_set[params['label']] == params['label_categories'][1]].pm10.isnull().sum()

# Missing value imputation
validation_set.loc[validation_set[(validation_set[params['label']] == params['label_categories'][0]) & (validation_set['pm10'].isnull() == True)].index, "pm10"] = imputation_baik
validation_set.loc[validation_set[(validation_set[params['label']] == params['label_categories'][1]) & (validation_set['pm10'].isnull() == True)].index, "pm10"] = imputation_tidak_sehat

# Checking mising value proportion
validation_set[validation_set[params['label']] == params['label_categories'][0]].pm10.isnull().sum(), validation_set[validation_set[params['label']] == params['label_categories'][1]].pm10.isnull().sum()

# Checking mising value proportion
testing_set[testing_set[params['label']] == params['label_categories'][0]].pm10.isnull().sum(), testing_set[testing_set[params['label']] == params['label_categories'][1]].pm10.isnull().sum()

# Missing value imputation
testing_set.loc[testing_set[(testing_set[params['label']] == params['label_categories'][0]) & (testing_set['pm10'].isnull() == True)].index, "pm10"] = imputation_baik
testing_set.loc[testing_set[(testing_set[params['label']] == params['label_categories'][1]) & (testing_set['pm10'].isnull() == True)].index, "pm10"] = imputation_tidak_sehat

# Checking mising value proportion
testing_set[testing_set[params['label']] == params['label_categories'][0]].pm10.isnull().sum(), testing_set[testing_set[params['label']] == params['label_categories'][1]].pm10.isnull().sum()

# Calculate mean
imputation_baik = int(train_set[train_set[params['label']] == params['label_categories'][0]].pm25.mean())
imputation_tidak_sehat = int(train_set[train_set[params['label']] == params['label_categories'][1]].pm25.mean())

# Show mean
imputation_baik, imputation_tidak_sehat

# Checking mising value proportion
train_set[train_set[params['label']] == params['label_categories'][0]].pm25.isnull().sum(), train_set[train_set[params['label']] == params['label_categories'][1]].pm25.isnull().sum()

# Missing value imputation
train_set.loc[train_set[(train_set[params['label']] == params['label_categories'][0]) & (train_set['pm25'].isnull() == True)].index, "pm25"] = imputation_baik
train_set.loc[train_set[(train_set[params['label']] == params['label_categories'][1]) & (train_set['pm25'].isnull() == True)].index, "pm25"] = imputation_tidak_sehat

# Checking mising value proportion
train_set[train_set[params['label']] == params['label_categories'][0]].pm25.isnull().sum(), train_set[train_set[params['label']] == params['label_categories'][1]].pm25.isnull().sum()

# Checking mising value proportion
validation_set[validation_set[params['label']] == params['label_categories'][0]].pm25.isnull().sum(), validation_set[validation_set[params['label']] == params['label_categories'][1]].pm25.isnull().sum()

# Missing value imputation
validation_set.loc[validation_set[(validation_set[params['label']] == params['label_categories'][0]) & (validation_set['pm25'].isnull() == True)].index, "pm25"] = imputation_baik
validation_set.loc[validation_set[(validation_set[params['label']] == params['label_categories'][1]) & (validation_set['pm25'].isnull() == True)].index, "pm25"] = imputation_tidak_sehat

# Checking mising value proportion
validation_set[validation_set[params['label']] == params['label_categories'][0]].pm25.isnull().sum(), validation_set[validation_set[params['label']] == params['label_categories'][1]].pm25.isnull().sum()

# Checking mising value proportion
testing_set[testing_set[params['label']] == params['label_categories'][0]].pm25.isnull().sum(), testing_set[testing_set[params['label']] == params['label_categories'][1]].pm25.isnull().sum()

# Missing value imputation
testing_set.loc[testing_set[(testing_set[params['label']] == params['label_categories'][0]) & (testing_set['pm25'].isnull() == True)].index, "pm25"] = imputation_baik
testing_set.loc[testing_set[(testing_set[params['label']] == params['label_categories'][1]) & (testing_set['pm25'].isnull() == True)].index, "pm25"] = imputation_tidak_sehat

# Checking mising value proportion
testing_set[testing_set[params['label']] == params['label_categories'][0]].pm25.isnull().sum(), testing_set[testing_set[params['label']] == params['label_categories'][1]].pm25.isnull().sum()

# Calculate mean and median
imputation_so2 = int(train_set['so2'].mean())
imputation_co = int(train_set['co'].median())
imputation_o3 = int(train_set['o3'].median())
imputation_no2 = int(train_set['no2'].median())

# Store into dictionary
imputation_values = {"so2" : imputation_so2, "co" : imputation_co, "o3" : imputation_o3, "no2" : imputation_no2}

# Show dictionary
imputation_values

# Handling missing value

# Train set
train_set.fillna(value = imputation_values, inplace = True)

# Validation set
validation_set.fillna(value = imputation_values, inplace = True)

# Test set
testing_set.fillna(value = imputation_values, inplace = True)

print(f"Train set \n{train_set.isna().sum()}, \n------------- \nValidation set \n{validation_set.isna().sum()}, \n------------- \nTest set \n{testing_set.isna().sum()}")

# Check proportion of stasiun
train_set['stasiun'].value_counts()

# Check proportion of stasiun
testing_set['stasiun'].value_counts()

# Check proportion of stasiun
validation_set['stasiun'].value_counts()

# Sanity check
print(train_set.shape, validation_set.shape, testing_set.shape)

# In[59]:


# Create numerical (num) and categorical (cat) split function
def num_cat_split(df, categorical_column):
    
    """
    This function is created to split categorical and numeric column
    :param df: <pandas dataframe> data contain sample
    :param categorical_column: <string> categorical column name
    :return categorical_data: <pandas dataframe> categorical data
    :return numerical_data: <pandas dataframe> numerical data
    :return categorical_ohe: <pandas dataframe> categorical data applied one hot encoding 
    """
    categorical_data = df[categorical_column]
    numerical_data = df.drop(categorical_column, axis = 1)
    categorical_ohe = pd.get_dummies(categorical_data)
      
    return categorical_data, numerical_data, categorical_ohe

# One Hot Encoding
categorical_data_train, numerical_data_train, categorical_ohe_train = num_cat_split(train_set, 'stasiun')
categorical_data_valid, numerical_data_valid, categorical_ohe_valid = num_cat_split(validation_set, 'stasiun')
categorical_data_test, numerical_data_test, categorical_ohe_test = num_cat_split(testing_set, 'stasiun')

# Concat
train_set = pd.concat([categorical_ohe_train, numerical_data_train], axis = 1)
validation_set = pd.concat([categorical_ohe_valid, numerical_data_valid], axis = 1)
testing_set = pd.concat([categorical_ohe_test, numerical_data_test], axis = 1)

# Sanity check
print(train_set.shape, validation_set.shape, testing_set.shape)

# Checking proportion
train_set[params['label']].value_counts()

# Checking proportion
validation_set[params['label']].value_counts()

# Checking proportion
testing_set[params['label']].value_counts()

# Replace value
replace_value(data = train_set, 
              column = params['label'],
              column_number = "single",
              old_value = "BAIK",
              replace_value = 1)

# Replace value
replace_value(data = train_set, 
              column = params['label'],
              column_number = "single",
              old_value = "TIDAK SEHAT",
              replace_value = 0)

# Replace value
replace_value(data = validation_set, 
              column = params['label'],
              column_number = "single",
              old_value = "BAIK",
              replace_value = 1)

# Replace value
replace_value(data = validation_set, 
              column = params['label'],
              column_number = "single",
              old_value = "TIDAK SEHAT",
              replace_value = 0)

# Replace value
replace_value(data = testing_set, 
              column = params['label'],
              column_number = "single",
              old_value = "BAIK",
              replace_value = 1)

# Replace value
replace_value(data = testing_set, 
              column = params['label'],
              column_number = "single",
              old_value = "TIDAK SEHAT",
              replace_value = 0)


# Create function to drop specified column
def drop_column(df,columns):
    
    """
    This function is used to drop specified column within dataframe
    
    :param df: <pandas dataframe> data contain sample
    :param columns: <string> column name
    :return churn_data: <pandas dataframe> data contain sample
    """
    
    dataset = df.drop(columns, axis = 1)
    
    return dataset

# Drop columns
train_set = drop_column(train_set, ['tanggal','max', 'critical'])
validation_set = drop_column(validation_set, ['tanggal','max', 'critical'])
testing_set = drop_column(testing_set, ['tanggal', 'max', 'critical'])

# Sanity check
print(train_set.shape, validation_set.shape, testing_set.shape)

# Undersampling
rus = RandomUnderSampler(random_state = 123)

# Data splitting
X_rus, y_rus = rus.fit_resample(train_set.drop(params['label'], axis = 1), train_set[params['label']])

# Data combined
train_set_rus = pd.concat([X_rus, y_rus], axis = 1)

# Checking proportion using plot
sns.histplot(train_set_rus, x = params['label'], hue = params['label'])

# Checking proportion
train_set_rus[params['label']].value_counts()

# Oversampling
ros = RandomOverSampler(random_state = 123)

# Data splitting
X_ros, y_ros = ros.fit_resample(train_set.drop(params['label'], axis = 1), train_set[params['label']])

# Data combined
train_set_ros = pd.concat([X_ros, y_ros], axis = 1)

# Checking proportion using plot
sns.histplot(train_set_ros, x = params['label'], hue = params['label'])

# Checking proportion
train_set_ros[params['label']].value_counts()

# SMOTE
sm = SMOTE(random_state = 123)

# Data splitting
X_sm, y_sm = sm.fit_resample(train_set.drop(params['label'], axis = 1), train_set[params['label']])

# Data combined
train_set_sm = pd.concat([X_sm, y_sm], axis = 1)

# Checking proportion using plot
sns.histplot(train_set_sm, x = params['label'], hue = params['label'])

# Checking proportion
train_set_sm[params['label']].value_counts()

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

# Input output split
X_train_set, y_train_set = input_output_split(df = train_set, column = params['label'])

# cat num split
X_train_cat, X_train_num, X_train_OHE = num_cat_split(X_train_set, params['range_stasiun'])

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
kendall_rank(input = X_train_num, output = y_train_set)

# Create contingency table and chi-square calculation function
def chi_square_test(input, output):
    
    """
    This function is created to calculate the level of significance between object type input data and object type 
    output data using chi-squared test
    
    :param input: <pandas dataframe> data contain input
    :param output: <pandas dataframe> data contain output
    :return contingency_tables: <dictionary> collection of distribution of a set of categorical variables
    :return p_values_chi2: <dictionary> collection of p-value
    """
    
    # Create an empty dictionary to store the contingency tables
    contingency_tables = {}
    
    # Loop over each column in the boolean or categorical DataFrame
    for col in input.columns:
        
        # Create the contingency table
        contingency_table = pd.crosstab(output, input[col])
        
        # Store the contingency table in the dictionary
        contingency_tables[col] = contingency_table
        
    # Print the contingency tables
    for col, contingency_table in contingency_tables.items():
        print(f"{col}:")
        print(contingency_table)

    print("----------")    
        
    # Create an empty dictionary to store the p-values
    p_values_chi2 = {}
    
    # Loop over each contingency table
    for col, contingency_table in contingency_tables.items():
        
        # Calculate the chi-squared test statistic and p-value
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Store the p-value in the dictionary
        p_values_chi2[col] = p
        
    # Print the p-values
    for col, p in p_values_chi2.items():
        print(f"{col}: {p:.7f}")

# Show contingency table and p_value
chi_square_test(input = X_train_cat, output = y_train_set)

# Probability of each station
for col in params['range_stasiun']:
    prob = X_train_set[col].sum() / len(X_train_set)
    print(f"{col} = {prob}")

# Split to subset
train_set_1 = train_set[train_set[params['range_stasiun'][0]] != 0]
train_set_2 = train_set[train_set[params['range_stasiun'][1]] != 0]
train_set_3 = train_set[train_set[params['range_stasiun'][2]] != 0]
train_set_4 = train_set[train_set[params['range_stasiun'][3]] != 0]
train_set_5 = train_set[train_set[params['range_stasiun'][4]] != 0]

# Sanity check
print(train_set_1.shape, train_set_2.shape, train_set_3.shape, train_set_4.shape, train_set_5.shape)

# Input output split
X_train_set_1, y_train_set_1 = input_output_split(df = train_set_1, column = params['label'])
X_train_set_2, y_train_set_2 = input_output_split(df = train_set_2, column = params['label'])
X_train_set_3, y_train_set_3 = input_output_split(df = train_set_3, column = params['label'])
X_train_set_4, y_train_set_4 = input_output_split(df = train_set_4, column = params['label'])
X_train_set_5, y_train_set_5 = input_output_split(df = train_set_5, column = params['label'])

# cat num split
X_train_cat_1, X_train_num_1, X_train_OHE_1 = num_cat_split(X_train_set_1, params['range_stasiun'])
X_train_cat_2, X_train_num_2, X_train_OHE_2 = num_cat_split(X_train_set_2, params['range_stasiun'])
X_train_cat_3, X_train_num_3, X_train_OHE_3 = num_cat_split(X_train_set_3, params['range_stasiun'])
X_train_cat_4, X_train_num_4, X_train_OHE_4 = num_cat_split(X_train_set_4, params['range_stasiun'])
X_train_cat_5, X_train_num_5, X_train_OHE_5 = num_cat_split(X_train_set_5, params['range_stasiun'])

# Show Kendall's rank coefficient and p-value
kendall_rank(input = X_train_num_1, output = y_train_set_1)

# Show Kendall's rank coefficient and p-value
kendall_rank(input = X_train_num_2, output = y_train_set_2)

# Show Kendall's rank coefficient and p-value
kendall_rank(input = X_train_num_3, output = y_train_set_3)

# Show Kendall's rank coefficient and p-value
kendall_rank(input = X_train_num_4, output = y_train_set_4)

# Show Kendall's rank coefficient and p-value
kendall_rank(input = X_train_num_5, output = y_train_set_5)

# Save as pickle
joblib.dump(X_train_set, "app/processed/X_train_feng.pkl")
joblib.dump(y_train_set, "app/processed/y_train_feng.pkl")

joblib.dump(X_rus, "app/processed/X_rus.pkl")
joblib.dump(y_rus, "app/processed/y_rus.pkl")

joblib.dump(X_ros, "app/processed/X_ros.pkl")
joblib.dump(y_ros, "app/processed/y_ros.pkl")

joblib.dump(X_sm, "app/processed/X_sm.pkl")
joblib.dump(y_sm, "app/processed/y_sm.pkl")

joblib.dump(validation_set.drop(columns = "categori"), "app/processed/X_valid_feng.pkl")
joblib.dump(validation_set.categori, "app/processed/y_valid_feng.pkl")

joblib.dump(testing_set.drop(columns = "categori"), "app/processed/X_test_feng.pkl")
joblib.dump(testing_set.categori, "app/processed/y_test_feng.pkl")


import streamlit as st
import pandas as pd
import requests
import time
import joblib
import yaml
import data_pipeline as pipeline
import preprocessing_and_feature_engineering as preprocess

def load_parameter(parameter_direction):
    with open(parameter_direction,'r') as file:
        params = yaml.safe_load(file)
    
    return params

def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)

# Load config
config = load_parameter("app/config/configuration_file_1.yaml")
model = pickle_load("app/models/production_model")

# Add some information about the service
st.title("Air Contaminant Standard Index Prediction")
st.subheader("The Prediction will utilised Machine Learning. Please Enjoy!")

# Create form of input
with st.form(key = "air_data_form"):
    # Create select box input
    stasiun = st.selectbox(
        label = "1.\tFrom which station is this data collected?",
        options = (
            "DKI1 (Bunderan HI)",
            "DKI2 (Kelapa Gading)",
            "DKI3 (Jagakarsa)",
            "DKI4 (Lubang Buaya)",
            "DKI5 (Kebon Jeruk) Jakarta Barat"
        )
    )

    # Create box for number input
    pm10 = st.number_input(
        label = "2.\tEnter PM10 Value:",
        min_value = 0,
        max_value = 800,
        help = "Value range from 0 to 800"
    )
    
    pm25 = st.number_input(
        label = "3.\tEnter PM25 Value:",
        min_value = 0,
        max_value = 400,
        help = "Value range from 0 to 400"
    )

    so2 = st.number_input(
        label = "4.\tEnter SO2 Value:",
        min_value = 0,
        max_value = 500,
        help = "Value range from 0 to 500"
    )

    co = st.number_input(
        label = "5.\tEnter CO Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )

    o3 = st.number_input(
        label = "6.\tEnter O3 Value:",
        min_value = 0,
        max_value = 160,
        help = "Value range from 0 to 160"
    )

    no2 = st.number_input(
        label = "7.\tEnter NO2 Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "stasiun": [stasiun],
            "pm10": [pm10],
            "pm25": [pm25],
            "so2": [so2],
            "co": [co],
            "o3": [o3],
            "no2": [no2]
            }

        # animation     
        with st.spinner('wait for it...'):
            time.sleep(3)
        st.balloons()        

        # Convert data api to dataframe
        data = pd.DataFrame(raw_data)

        # Convert dtype
        data = pd.concat(
            [
                data[config["predictors"][0]],
                data[config["predictors"][1:]].astype(int)
            ],
            axis = 1
        )

        # Create empty dictionary
        result = {}      
    
        try:
            pipeline.check_data(data, config, True)
        except AssertionError as ae:
            result['prediction'] = {"res": [], "error_msg": str(ae)}

        # Encoding stasiun
        categorical_data, numerical_data, categorical_ohe = preprocess.num_cat_split(data, "stasiun")

        # Adding missing station
        missing_categories = set(config['range_stasiun']) - set(categorical_ohe.columns)
        for category in missing_categories:
            categorical_ohe[category] = 0

        # Define the desired order of column
        station_order = ['DKI1 (Bunderan HI)', 'DKI2 (Kelapa Gading)', 'DKI3 (Jagakarsa)', 'DKI4 (Lubang Buaya)', 'DKI5 (Kebon Jeruk) Jakarta Barat']
        predictor_order = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']

        # Reorder columns to match desired station order
        categorical_ohe = categorical_ohe[station_order]
        numerical_data = numerical_data[predictor_order]

        # Concat data
        data = pd.concat([categorical_ohe,numerical_data], axis=1)

        # Predict data
        y_pred = model.predict(data)

        # Inverse transform
        if 'prediction' not in result:
            if y_pred[0] == 1:
                result['prediction'] = "Good Air Quality"
            else:
                result['prediction'] = "Bad Air Quality"
    
        # Print or use the result variable as needed
        st.success(result['prediction'])
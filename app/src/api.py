import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import yaml
import joblib
from pydantic import BaseModel
import data_pipeline as pipeline
import preprocessing_and_feature_engineering as preprocess
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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

# Create class describes predictor variables
class air_quality(BaseModel):
    stasiun : str
    pm10 : int
    pm25 : int
    so2 : int
    co : int
    o3 : int
    no2 : int

# Create the app object
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, stranger'}

@app.get('/{name}')
def get_name(name: str):
    return {"Welcome to air quality predictor homepage": f"{name}"}

@app.post("/predict/")
def predict(data: air_quality):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
    
    # Convert dtype
    data = pd.concat(
        [
            data[config["predictors"][0]],
            data[config["predictors"][1:]].astype(int)
        ],
        axis = 1
    )

    # Check range data
    try:
        pipeline.check_data(data, config, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
     
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
    
    # Inverse tranform
    if(y_pred[0] == 1):
        y_pred = "Good Air Quality"
    else:
        y_pred = "Bad Air Quality"
    return {'prediction': y_pred}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8000)
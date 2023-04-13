import pickle
from urllib import request
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the API!"}

# Define endpoint for making predictions
# @app.post('/predictsex')
# def predict(data:dict):
#   # Load model from .pkl file
#   with open('./MF_XGB_XV2.pkl','rb') as file:
#     model = pickle.load(file)
#     # Convert the DataFrame to a DMatrix object
#     d_input = xgb.DMatrix(data)

#     # Make the prediction using the XGBoost model
#     prediction = model.predict(d_input)
#     # Convert input data to DataFrame
#     # df = pd.DataFrame(data, index=[0])
#     # # Make prediction
#     # prediction = model.predict(df)
#     # Return Prediction as JSON response
#     return {'prediction': prediction[0]}

import numpy as np
import scipy.sparse as sp
import xgboost as xgb

# @app.post('/predictsex')
# def predict(data:dict):
#   # Load model from .pkl file
#   with open('./MF_XGB_XV2.pkl','rb') as file:
#     model = pickle.load(file)
#     # Convert the dictionary input to a sparse matrix
#     row = np.zeros(len(data))
#     col = np.arange(len(data))
#     data_sparse = sp.coo_matrix((row, (row, col)), shape=(1, len(data)))
#     for key, value in data.items():
#         if isinstance(value, str):
#             value = [value]
#         data_sparse[0, col[key]] = value
#     d_input = xgb.DMatrix(data_sparse)

#     # Make the prediction using the XGBoost model
#     prediction = model.predict(d_input)
#     # Return Prediction as JSON response
#     return {'prediction': prediction[0]}



# new sex one
  #second one

  # Define endpoint for making predictions
@app.post('/predictwrapping')
def predict(data:dict):
  # Load model from .pkl file
  with open('./final_wrappingXGB_XV2.pkl','rb') as file:
    model = pickle.load(file)

    # Convert the DataFrame to a DMatrix object
    d_input = xgb.DMatrix(data)

    # Make the prediction using the XGBoost model
    prediction = model.predict(d_input)
    # Convert input data to DataFrame
    # df = pd.DataFrame(data, index=[0])
    # # Make prediction
    # prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}
  


import jsonify


@app.post('/predict')
def predict():
    with open('./MF_XGB_XV2.pkl','rb') as file:
        model = pickle.load(file)
    # Get the input data as a JSON object
    data = request.get_json()
    # Check if the Content-Type header is set to "application/json"
    content_type = request.headers.get('Content-Type')
    if content_type != 'application/json':
        return jsonify({'error': 'Invalid Content-Type header'}), 400
    # Convert the input data into a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index').T
    # Convert "depth" and "length" columns to floats
    df["depth"] = df["depth"].astype(float)
    df["length"] = df["length"].astype(float)
    # Convert categorical variables to category data type
    cat_cols = ["headdirection", "depth", "facebundles",
                'goods', 'wrapping', 'haircolor', 'samplescollected', 'length', 'ageatdeath']
    for col in cat_cols:
        df[col] = df[col].astype("category")
    # Reorder the columns in the DataFrame to match the order of the features in the XGBoost model
    df = df[["headdirection", "depth", "facebundles",
             'goods', 'wrapping', 'haircolor', 'samplescollected', 'length', 'ageatdeath']]
    # Make predictions using the XGBoost model
    dtest = xgb.DMatrix(df, enable_categorical=True)
    y_pred = model.predict(dtest)
    # Convert the output from 1/0 to "male"/"female"
    result = []
    for p in y_pred:
        if p == 1:
            result.append("male")
        else:
            result.append("female")

    return jsonify(predictions=result)






from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    headdirection: str = ...
    depth: float = ...
    facebundles: str = ...
    goods: str = ...
    wrapping: str = ...
    haircolor: str = ...
    samplescollected: str = ...
    length: float = ...
    ageatdeath: str = ...

@app.post('/predictsex_tryme')
def predict(input_data: InputData):
    return "Predicted sex: Male"

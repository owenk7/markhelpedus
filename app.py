import pickle
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.post('/predictsex')
def predict(data:dict):
  # Load model from .pkl file
  with open('./MF_XGB_XV2.pkl','rb') as file:
    model = pickle.load(file)
    # Convert the dictionary input to a sparse matrix
    row = np.zeros(len(data))
    col = np.arange(len(data))
    data_sparse = sp.coo_matrix((row, (row, col)), shape=(1, len(data)))
    for key, value in data.items():
        if isinstance(value, str):
            value = [value]
        data_sparse[0, col[key]] = value
    d_input = xgb.DMatrix(data_sparse)

    # Make the prediction using the XGBoost model
    prediction = model.predict(d_input)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}



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
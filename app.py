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

from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd

app = FastAPI()

# Load the saved model
with open("DTCmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Define the predict endpoint
@app.post("/predict")
async def predict(input_data: dict):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


with open("DTCmodelSEX.pkl", "rb") as f:
    model = pickle.load(f)

# Define the predict endpoint
@app.post("/predictsex")
async def predict(input_data: dict):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
import pickle
import pandas as pd
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
@app.post('/predictsex')
def predict(data:dict):
  # Load model from .pkl file
  with open('./MF_XGB_XV2.pkl','rb') as file:
    model = pickle.load(file)
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Make prediction
    prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}
  

  #second one

  # Define endpoint for making predictions
@app.post('/predictwrapping')
def predict(data:dict):
  # Load model from .pkl file
  with open('./final_wrappingXGB_XV2.pkl','rb') as file:
    model = pickle.load(file)
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Make prediction
    prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}
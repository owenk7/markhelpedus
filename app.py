import pickle
from urllib import request
import pandas as pd
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

from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd

# Load the saved model
with open("DTCmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Define the predict endpoint
@app.post("/predict")
async def predict(input_data: dict):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        label_map = {"B": "Bones (No Wrapping)", "H": "Half Wrap", "W": "Whole Wrap"}
        label = label_map[prediction[0]]
        return {"prediction": label}    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


with open("DTCmodelSEX.pkl", "rb") as f:
    model2 = pickle.load(f)

# Define the predict endpoint
@app.post("/predictsex")
async def predict(input_data2: dict):
    try:
        input_df2 = pd.DataFrame([input_data2])
        prediction2 = model2.predict(input_df2)
        label2 = "Male" if prediction2[0] == "M" else "Female"
        return {"prediction": label2}    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
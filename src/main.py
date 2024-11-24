# main.py

from fastapi import FastAPI, File, UploadFile
from model import train_model
from prediction import predict
import pandas as pd
import os

app = FastAPI()

# Define file paths
DATA_PATH = "uploaded_data.csv"
MODEL_PATH = "music_popularity_model.pkl"

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Music Popularity Prediction API"}

@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    """Endpoint for making predictions."""
    # Save uploaded file
    with open(DATA_PATH, "wb") as f:
        f.write(await file.read())

    # Make predictions
    predictions = predict(DATA_PATH, MODEL_PATH)
    return {"predictions": predictions.tolist()}

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """Endpoint for retraining the model with new data."""
    # Save uploaded file
    with open(DATA_PATH, "wb") as f:
        f.write(await file.read())

    # Retrain the model
    model, _, _ = train_model(DATA_PATH, MODEL_PATH)
    return {"message": "Model retrained and updated successfully."}

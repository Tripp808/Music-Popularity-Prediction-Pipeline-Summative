# main.py

from fastapi import FastAPI, File, UploadFile
from model import train_model
from prediction import predict
from uuid import uuid4  # To generate unique filenames
import os

app = FastAPI()

# Define file paths
MODEL_PATH = "music_popularity_model.pkl"


def save_uploaded_file(file: UploadFile):
    """Save the uploaded file with a unique name."""
    unique_filename = f"uploaded_{uuid4().hex}.csv"
    with open(unique_filename, "wb") as f:
        f.write(file.file.read())
    return unique_filename


def delete_file(file_path):
    """Delete a file after processing."""
    if os.path.exists(file_path):
        os.remove(file_path)


@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    """Endpoint for making predictions."""
    # Save uploaded file with a unique name
    unique_filename = save_uploaded_file(file)

    try:
        # Make predictions
        predictions = predict(unique_filename, MODEL_PATH)
        return {"predictions": predictions.tolist()}
    finally:
        # Clean up the file after processing
        delete_file(unique_filename)


@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """Endpoint for retraining the model with new data."""
    # Save uploaded file with a unique name
    unique_filename = save_uploaded_file(file)

    try:
        # Retrain the model
        model, _, _ = train_model(unique_filename, MODEL_PATH)
        return {"message": "Model retrained and updated successfully."}
    finally:
        # Clean up the file after processing
        delete_file(unique_filename)

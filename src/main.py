from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import train_model
from prediction import predict
from uuid import uuid4  # for unique filenames
import os

app = FastAPI()

# Define file paths
MODEL_PATH = "music_popularity_model.pkl"

# CORS middleware to allow requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any domain, i will update with domain name after i deploy
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


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


@app.get("/")
async def root():
    """Root endpoint providing API info."""
    return {
        "message": "Welcome to the Music Popularity Predictor API!",
        "endpoints": {
            "/predict": "Upload a CSV file to get predictions",
            "/retrain": "Upload a CSV file to retrain the model",
        },
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    """Endpoint for making predictions."""
    # Save uploaded file with a unique name
    unique_filename = save_uploaded_file(file)

    try:
        # Make predictions and generate visualization
        predictions_df, visualization = predict(unique_filename, MODEL_PATH)

        predictions = predictions_df.to_dict(orient="records")

        return {"predictions": predictions, "visualization": visualization}
    finally:
        # Clean up the file after processing
        delete_file(unique_filename)


@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    """Endpoint for retraining the model with new data."""
    # Save with unique name
    unique_filename = save_uploaded_file(file)

    try:
        # Retrain the model
        model, _, _ = train_model(unique_filename, MODEL_PATH)
        return {"message": "Model retrained and updated successfully."}
    finally:
        # Clean up the file after processing
        delete_file(unique_filename)

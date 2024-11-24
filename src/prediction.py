import pickle
import pandas as pd
from preprocessing import preprocess_data

def load_model(model_path='music_popularity_model.pkl'):
    """Loads a pre-trained model."""
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def predict(file_path, model_path='music_popularity_model.pkl'):
    """Preprocesses new data and makes predictions."""
    model = load_model(model_path)
    X, _ = preprocess_data(file_path)
    predictions = model.predict(X)
    return predictions

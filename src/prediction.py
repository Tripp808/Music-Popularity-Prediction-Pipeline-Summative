import pickle
import pandas as pd
from preprocessing import preprocess_data

def load_model(model_path='music_popularity_model.pkl'):
    """Loads a pre-trained model."""
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"Model loaded successfully from {model_path}.")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def predict(file_path, model_path='music_popularity_model.pkl'):
    """Preprocesses new data and makes predictions."""
    try:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)

        print(f"Preprocessing data from {file_path}...")
        X, _ = preprocess_data(file_path)
        
        # Validate preprocessed data
        if X is None or len(X) == 0:
            raise ValueError("Preprocessed data is empty or invalid.")
        print(f"Preprocessed data shape: {X.shape}")

        # Generate predictions
        predictions = model.predict(X)
        print(f"Predictions generated: {predictions}")
        return predictions
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except ValueError as ve:
        raise ValueError(f"Value error during prediction: {str(ve)}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during prediction: {str(e)}")

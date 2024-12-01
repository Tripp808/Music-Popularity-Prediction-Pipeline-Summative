import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
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
    """Preprocesses new data, makes predictions, and generates visualizations."""
    try:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)

        print(f"Reading data from {file_path}...")
        data = pd.read_csv(file_path)

        # Ensure necessary columns exist
        if not {"Track", "Artist"}.issubset(data.columns):
            raise ValueError("The CSV file must contain 'Track' and 'Artist' columns.")

        print("Preprocessing data...")
        X, _ = preprocess_data(file_path)

        # Validate preprocessed data
        if X is None or len(X) == 0:
            raise ValueError("Preprocessed data is empty or invalid.")
        print(f"Preprocessed data shape: {X.shape}")

        # Generate predictions
        predictions = model.predict(X)
        print(f"Predictions generated: {predictions}")

        # Combine predictions with Track and Artist columns
        data["Prediction"] = predictions
        result = data[["Track", "Artist", "Prediction"]]

        # 1. Visualization: Distribution of Popularity (Prediction)
        plt.figure(figsize=(8, 6))
        data["Prediction"].value_counts().plot(kind="bar", color=["#4CAF50", "#FF5252"])
        plt.title("Distribution of Popularity Prediction")
        plt.xlabel("Prediction")
        plt.ylabel("Number of Songs")
        plt.xticks([0, 1], ["Will Perform Well", "Will Not Perform Well"], rotation=0)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        visualization_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()
        
        # 2. Visualization: Correlation Heatmap (Numerical Features)
        numeric_data = data.select_dtypes(include=["float64", "int64"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap of Numerical Features")
        plt.tight_layout()
        correlation_buffer = io.BytesIO()
        plt.savefig(correlation_buffer, format="png")
        correlation_buffer.seek(0)
        correlation_base64 = base64.b64encode(correlation_buffer.getvalue()).decode("utf-8")
        correlation_buffer.close()

        # 3. Visualization: Feature Importance (Bar Chart)
        # Feature importance visualization requires the model to support it (e.g., tree-based models)
        try:
            feature_importances = model.feature_importances_  # Example for tree-based models
            features = X.columns
            importance_data = pd.DataFrame({
                "Feature": features,
                "Importance": feature_importances
            }).sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=importance_data, palette="viridis")
            plt.title("Feature Importance")
            plt.tight_layout()
            importance_buffer = io.BytesIO()
            plt.savefig(importance_buffer, format="png")
            importance_buffer.seek(0)
            importance_base64 = base64.b64encode(importance_buffer.getvalue()).decode("utf-8")
            importance_buffer.close()
        except AttributeError:
            importance_base64 = None  # Model does not support feature importance

        print("Visualizations generated successfully.")
        visualizations = {
            "prediction_distribution": visualization_base64,
            "correlation_heatmap": correlation_base64,
            "feature_importance": importance_base64
        }

        return result, visualizations
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except ValueError as ve:
        raise ValueError(f"Value error during prediction: {str(ve)}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during prediction: {str(e)}")

from locust import HttpUser, task
import os

class MLUser(HttpUser):
    # Path to the dataset for prediction and retraining
    predict_file_path = os.path.join("data", "train", "top_songs.csv")
    retrain_file_path = os.path.join("data", "test", "SonyMusic_songs.csv")

    @task
    def send_prediction(self):
        """Send a prediction request to the /predict endpoint."""
        with open(self.predict_file_path, "rb") as file:
            files = {"file": ("top_songs.csv", file, "text/csv")}
            # POST request to the /predict endpoint
            self.client.post("/predict", files=files)

    @task
    def retrain_model(self):
        """Send a retrain request to the /retrain endpoint."""
        with open(self.retrain_file_path, "rb") as file:
            files = {"file": ("SonyMusic_songs.csv", file, "text/csv")}
            # POST request to the /retrain endpoint
            self.client.post("/retrain", files=files)

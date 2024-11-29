from locust import HttpUser, task
import os

class MLUser(HttpUser):
    @task
    def send_prediction(self):
        file_path = os.path.join("data", "train", "top_songs.csv")
        
        with open(file_path, "rb") as file:
            files = {"file": ("top_songs.csv", file, "text/csv")}
            # POST request to the /predict endpoint
            self.client.post("/predict", files=files)

# SoundIQ: Music Popularity Prediction Web App

SoundIQ is a full-stack web application designed to predict the popularity of music tracks. The platform provides an interactive interface for users to upload datasets, retrain prediction models, and gain insights into music trends based on machine learning algorithms.

## 🔗 Links

- **Live Web App**: [SoundIQ on Vercel](https://sound-iq.vercel.app/)
- **Backend API**: [SoundIQ Backend on Render](https://music-popularity-predictor.onrender.com/) The APi is run from the main.py in the src folder and contains all the functions and endpoints including the other pipeline scripts used.
- **Demo Video**: [Loom Demo](https://www.loom.com/share/f78f850e01e340cc9484b34234fe8ebd?sid=544dd36e-fad9-48a0-ad9b-17649a679607)
- You can directly download and use any of the datasets for testing in the data/test directory of this repo. The original dataset that was usied for training is in data/train
- **Docker Image**: [oche99/music-predictor on Docker Hub](https://hub.docker.com/r/oche99/music-predictor) Run this docker pull oche99/music-predictor
- I created the docker-compose.yml file which is in my root to simulate different locust test for different containers. I ran this
  locust -f locustfile.py
- You can find the locust test results on different docker containers here: https://drive.google.com/drive/folders/1wMcjzRRAPnfHpebfsX7m5M1X-EDslAyX?usp=sharing

---

## 🌟 Project Description

SoundIQ simplifies the task of predicting music popularity using a machine learning pipeline. It features:

- A **frontend** built with React and styled using CSS for user interaction.
- A **backend** powered by FastAPI, which processes data and handles predictions.
- A machine learning model integrated into the backend for real-time predictions.
- Deployment on **Vercel** (frontend) and **Render** (backend), ensuring scalability and availability.

Key functionalities:

1. **Music Popularity Prediction**: Upload track data to get predictions.
2. **Model Retraining**: Update the prediction model with new datasets.
3. **Data Insights**: Interactive visualizations and result interpretation.

---

## 🚀 How to Set It Up

### Prerequisites

Ensure you have the following installed:

- **Node.js** (v16 or later)
- **Python** (v3.9 or later)
- **FastAPI** and related dependencies (listed in `requirements.txt`)
- **Docker** (for containerization)
- **Locust** (for load testing)

### 1. Clone the Repository

```bash
git clone https://github.com/Tripp808/soundIQ-webapp-ml-pipelines.git
cd soundIQ-webapp-ml-pipelines
```

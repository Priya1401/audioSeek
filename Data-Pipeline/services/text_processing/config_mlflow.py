import mlflow
import os

# Tracking server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "AudioSeek-Pipeline")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Ensure experiment exists
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
except Exception as e:
    print(f"Warning: Could not set MLflow experiment: {e}")

# AudioSeek Setup Guide

## 1. Prerequisites
- Docker & Kubernetes (Docker Desktop)
- Python 3.10+
- Google Cloud SDK (gcloud)

## 2. Environment Variables
The following files are **not** in the repository for security reasons. You must obtain them from the team lead or a secure password manager.

### `Data-Pipeline/.env`
Create this file in `Data-Pipeline/` with the following keys:
```bash
GCP_PROJECT_ID=ie7374-475102
GCP_BUCKET_NAME=audioseek-bucket
FIRESTORE_DB=(default)
GEMINI_API_KEY=your_gemini_api_key
```

### `frontend/.env.local`
Create this file in `frontend/` with the following keys:
```bash
API_URL=http://localhost:8001
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8501
```

## 3. GCP Credentials
1.  Obtain the `gcp-credentials.json` file.
2.  Place it in `Data-Pipeline/gcp-credentials.json`.

## 4. Running the Application

### Backend (Kubernetes)
```bash
# 1. Build the image
docker build --platform linux/amd64 -t us-east1-docker.pkg.dev/ie7374-475102/audioseek-repo/api:latest -f Data-Pipeline/Dockerfile Data-Pipeline

# 2. Push to Artifact Registry
docker push us-east1-docker.pkg.dev/ie7374-475102/audioseek-repo/api:latest

# 3. Deploy
kubectl apply -f k8s/api.yaml
```

### Frontend (Local)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

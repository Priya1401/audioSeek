#!/bin/bash
set -e

PROJECT_ID="ie7374-475102"
REGION="us-east1"
REPO_NAME="audioseek-repo"
CLUSTER_NAME="audioseek-cluster"

echo "=== 1. Building and Pushing API Image ==="
docker build --platform linux/amd64 -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/api:latest -f Data-Pipeline/Dockerfile Data-Pipeline
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/api:latest

echo "=== 2. Building and Pushing Frontend Image ==="
docker build --platform linux/amd64 -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/frontend:latest frontend
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/frontend:latest

echo "=== 3. Deploying Frontend to Cloud Run ==="
gcloud run deploy audioseek-frontend \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/frontend:latest \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated

echo "=== 4. Getting Cluster Credentials ==="
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID

echo "=== 5. Deploying Backend & MLflow to Kubernetes ==="
# Apply all manifests in k8s/ (api.yaml, mlflow.yaml)
kubectl apply -f k8s/

echo "=== Deployment Complete! ==="
echo "Frontend: Check Cloud Run URL above"
echo "Backend: kubectl get service audioseek-api"
echo "MLflow: kubectl get service mlflow"

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

echo "=== 3. Getting Cluster Credentials ==="
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID

echo "=== 4. Deploying to Kubernetes ==="
kubectl apply -f k8s/

echo "=== Deployment Complete! ==="
echo "Wait a minute for the LoadBalancer IP:"
echo "kubectl get service audioseek-frontend"

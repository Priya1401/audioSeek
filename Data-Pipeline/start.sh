#!/bin/bash

# Enable unbuffered output for Python (critical for Docker logs)
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Starting AudioSeek Services"
echo "=========================================="

# Start transcription service on port 8000
echo "Starting Transcription Service on port 8000..."
cd /app/services/transcription && uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info &

# Start text processing service on port 8001
echo "Starting Text Processing Service on port 8001..."
cd /app/services/text_processing && uvicorn main:app --host 0.0.0.0 --port 8001 --log-level info &

echo "=========================================="
echo "Both services started. Waiting..."
echo "=========================================="

# Wait for all background processes
wait
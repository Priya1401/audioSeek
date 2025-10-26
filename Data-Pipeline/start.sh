#!/bin/bash
cd /app/services/transcription && uvicorn main:app --host 0.0.0.0 --port 8000 &
cd /app/services/text_processing && uvicorn main:app --host 0.0.0.0 --port 8001 &
wait
# Text Processing Service

This service handles audio transcription, text chunking, embedding generation, and vector database management.

## Directory Structure

- **`api/`**: FastAPI application entry point (`main.py`) and route controllers (`controllers.py`).
- **`core/`**: Core configuration (`config.py`) and shared utilities (`utils.py`).
- **`domain/`**: Data models and schemas (`models.py`, `models_firestore.py`).
- **`services/`**: Business logic and external service integrations.
  - **`audio/`**: Audio processing services (Transcription).
  - **`nlp/`**: NLP services (Chunking, Embedding, QA, Pipeline).
  - **`storage/`**: Data storage services (Firestore, Vector DB, Metadata DB).
  - **`jobs/`**: Background job management.
  - **`notifications/`**: Notification services (Email).
- **`workers/`**: Background task workers (`tasks.py`).
- **`tests/`**: Unit and integration tests.

## Running the Service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

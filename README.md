# AudioSeek Audiobook Summarizer

A comprehensive MLOps pipeline for processing audiobooks into searchable, queryable knowledge bases using batch scripts, Airflow orchestration, and FastAPI microservices.

## Features

- **Batch Audio Transcription**: Process ZIP files of audio files using Faster-Whisper
- **Intelligent Chunking**: Segment transcripts into chapters and semantic chunks with entity extraction
- **Vector Embeddings**: Generate embeddings using Sentence-Transformers for semantic search
- **Vector Database**: Store and search embeddings using FAISS
- **Metadata Management**: SQLite database for structured audiobook metadata (chapters, entities, chunks)
- **Advanced QA**: Answer chapter-specific, cumulative, timestamp-based, and entity-focused questions using advanced LLM
- **Airflow Orchestration**: DAG-based pipeline for batch processing
- **Microservices**: FastAPI services for real-time processing
- **Containerization**: Docker and Docker Compose for easy deployment

## Architecture

### Components

- **Scripts (`Data-Pipeline/scripts/`)**: Batch processing modules for transcription, chunking, embedding, and validation
- **DAGs (`Data-Pipeline/dags/`)**: Airflow workflows for orchestrated pipeline execution
- **Services (`Data-Pipeline/services/`)**: FastAPI microservices for real-time operations
- **Data (`Data-Pipeline/data/`)**: Sample results, logs, and intermediate outputs

### Data Flow

1. **Batch Processing (Airflow)**: Audio ZIP → Transcription → Validation → Chunking → Embedding
2. **Real-Time Services**: Audio → Transcription Service → Text Processing Service (chunk/embed/store/QA)

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key
- SpaCy English model: `python -m spacy download en_core_web_sm`

## Installation and Running

### Docker Compose (Recommended)

```bash
cd audioSeek/Data-Pipeline
docker-compose up -d #d stands for detached mode
```

Access:
- Airflow UI: http://localhost:8080 (airflow2/airflow2)
- Transcription Service: http://localhost:8000
- Text Processing Service: http://localhost:8001


After you're done,
```bash
docker-compose down 

# check if any services are up
docker ps

# if there are any services up,
docker stop <container-id>
```

### Local Development

```bash
cd audioSeek/Data-Pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export API_KEY=your_llm_api_key

# Run services
cd services/transcription && uvicorn main:app --port 8000 &
cd services/text_processing && uvicorn main:app --port 8001 &
```

## Usage

### Batch Processing
1. Place audio ZIPs in shared volumes
2. Trigger "audio_processing_pipeline" DAG in Airflow UI


## Development

- Scripts: Batch processing with CLI
- DAGs: Airflow orchestration
- Services: FastAPI with MVC pattern
- MLOps: Containerization, logging, health checks


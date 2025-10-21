# AudioSeek Audiobook Summarizer

A comprehensive MLOps pipeline for processing audiobooks into searchable, queryable knowledge bases using microservices architecture.

## Features
 
- **Audio Transcription**: Convert audio files to timestamped text using Faster-Whisper
- **Intelligent Chunking**: Segment transcripts into chapters and semantic chunks with entity extraction
- **Vector Embeddings**: Generate embeddings using Sentence-Transformers for semantic search
- **Vector Database**: Store and search embeddings using FAISS
- **Metadata Management**: SQLite database for structured audiobook metadata (chapters, entities, chunks)
- **Advanced QA**: Answer chapter-specific, cumulative, timestamp-based, and entity-focused questions using OpenAI GPT
- **Full Pipeline**: End-to-end processing from audio upload to queryable system

## Architecture

### Microservices

- **Transcription Service** (port 8000): Audio-to-text with chapter detection
- **Chunking Service** (port 8003): Text segmentation with NLP enrichment
- **Embedding Service** (port 8001): Vector generation for semantic search
- **Vector DB Service** (port 8002): FAISS-based vector storage and retrieval
- **Metadata DB Service** (port 8006): SQLite for structured metadata
- **QA Service** (port 8004): Intelligent question answering with query parsing
- **Pipeline Service** (port 8005): Orchestrates full audiobook processing

### Database Schema

See `services/metadata_db/schema.sql` for the complete SQLite schema including:
- audiobooks, chapters, chunks, entities, entity_mentions tables
- Indexes for performance

## Prerequisites

- Python 3.10+
- OpenAI API key
- SpaCy English model: `python -m spacy download en_core_web_sm`

## Installation

1. Clone/download the project
2. Install dependencies:
   ```bash
   cd Data-Pipeline
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Set environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   ```

## Running the System

Run each service in separate terminals:

```bash
# Transcription Service
cd services/transcription && uvicorn main:app --host 0.0.0.0 --port 8000

# Chunking Service
cd services/chunking && uvicorn main:app --host 0.0.0.0 --port 8003

# Embedding Service
cd services/embedding && uvicorn main:app --host 0.0.0.0 --port 8001

# Vector DB Service
cd services/vector_db && uvicorn main:app --host 0.0.0.0 --port 8002

# Metadata DB Service
cd services/metadata_db && uvicorn main:app --host 0.0.0.0 --port 8006

# QA Service
cd services/qa && uvicorn main:app --host 0.0.0.0 --port 8004

# Pipeline Service
cd services/pipeline && uvicorn main:app --host 0.0.0.0 --port 8005
```

## Usage

### Process an Audiobook

Upload audio file to the pipeline service:

```bash
curl -X POST "http://localhost:8005/process_audiobook" \
  -F "file=@your_audiobook.mp3" \
  -F "title=Your Book Title" \
  -F "author=Author Name"
```

This will:
1. Transcribe the audio
2. Detect chapters
3. Chunk the text with entity extraction
4. Generate embeddings
5. Store everything in vector DB and metadata DB

### Ask Questions

Query the QA service with natural language questions:

```bash
curl -X POST "http://localhost:8004/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What happened in the second chapter?"}'
```

Supported query types:
- Chapter-specific: "What happened in chapter 2?"
- Cumulative: "What happened till chapter 16?"
- Entity-focused: "What happened to Adam in chapter 4?"
- Timestamp-based: "Who had a fight at 1:00:56?"

### API Endpoints

#### Transcription Service (8000)
- `POST /transcribe`: Upload audio file, returns transcript with chapters

#### Chunking Service (8003)
- `POST /chunk`: Process transcript, returns chunks, chapters, entities

#### Embedding Service (8001)
- `POST /embed`: Generate embeddings for text list

#### Vector DB Service (8002)
- `POST /add`: Store embeddings with metadata
- `POST /search`: Semantic search by embedding

#### Metadata DB Service (8006)
- CRUD endpoints for audiobooks, chapters, chunks, entities

#### QA Service (8004)
- `POST /ask`: Answer questions with citations

#### Pipeline Service (8005)
- `POST /process_audiobook`: Full pipeline processing

## MLOps Features

- **Logging**: Comprehensive logging in all services
- **Health Checks**: `/` and `/health` endpoints for monitoring
- **Reproducibility**: Versioned dependencies, schema migrations
- **Scalability**: Microservices architecture for horizontal scaling

## Development

- All services use FastAPI with Pydantic models
- Vector operations use FAISS for efficiency
- Metadata stored in SQLite for relational queries
- LLM integration via OpenAI API (easily replaceable with local models)

## Limitations & Future Improvements

- Vector DB is in-memory (add persistence for production)
- Single audiobook support (extend for multiple)
- Basic chapter detection (improve with ML models)
- OpenAI dependency (add local LLM support)

## License

[Add license information]
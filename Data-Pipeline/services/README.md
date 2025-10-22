# AudioSeek Audiobook Summarizer

A comprehensive MLOps pipeline for processing audiobooks into searchable, queryable knowledge bases using two main services.

## Features
 
- **Audio Transcription**: Convert audio files to timestamped text using Faster-Whisper
- **Intelligent Chunking**: Segment transcripts into chapters and semantic chunks with entity extraction
- **Vector Embeddings**: Generate embeddings using Sentence-Transformers for semantic search
- **Vector Database**: Store and search embeddings using FAISS
- **Metadata Management**: SQLite database for structured audiobook metadata (chapters, entities, chunks)
- **Advanced QA**: Answer chapter-specific, cumulative, timestamp-based, and entity-focused questions using OpenAI GPT
- **Full Pipeline**: End-to-end processing from audio upload to queryable system

## Architecture

### Services

- **Transcription Service** (port 8000): Audio-to-text with chapter detection (MVC pattern)
- **Text Processing Service** (port 8001): All downstream text processing, storage, and QA (MVC pattern)

### Database Schema

The SQLite database schema includes the following tables:

#### audiobooks
- `id` (INTEGER PRIMARY KEY): Unique identifier
- `title` (TEXT NOT NULL): Audiobook title
- `author` (TEXT): Author name
- `duration` (REAL): Duration in seconds
- `created_at` (TIMESTAMP): Creation timestamp

#### chapters
- `id` (INTEGER PRIMARY KEY): Unique identifier
- `audiobook_id` (INTEGER): Foreign key to audiobooks
- `title` (TEXT): Chapter title
- `start_time` (REAL): Start time in seconds
- `end_time` (REAL): End time in seconds
- `summary` (TEXT): Optional chapter summary

#### chunks
- `id` (INTEGER PRIMARY KEY): Unique identifier
- `audiobook_id` (INTEGER): Foreign key to audiobooks
- `chapter_id` (INTEGER): Foreign key to chapters
- `start_time` (REAL): Start time in seconds
- `end_time` (REAL): End time in seconds
- `text` (TEXT NOT NULL): Chunk text content
- `token_count` (INTEGER): Number of tokens in chunk
- `embedding_id` (INTEGER): Reference to FAISS index

#### entities
- `id` (INTEGER PRIMARY KEY): Unique identifier
- `name` (TEXT NOT NULL): Entity name
- `type` (TEXT): Entity type (character, place, etc.)
- `audiobook_id` (INTEGER): Foreign key to audiobooks

#### entity_mentions
- `id` (INTEGER PRIMARY KEY): Unique identifier
- `entity_id` (INTEGER): Foreign key to entities
- `chunk_id` (INTEGER): Foreign key to chunks
- `start_pos` (INTEGER): Start position in text
- `end_pos` (INTEGER): End position in text

#### Indexes
- `idx_chunks_audiobook` on chunks(audiobook_id)
- `idx_chunks_chapter` on chunks(chapter_id)
- `idx_entities_audiobook` on entities(audiobook_id)

The complete schema is in `services/text_processing/schema.sql`.

## Prerequisites

- Python 3.10+
- OpenAI API key
- SpaCy English model: `python -m spacy download en_core_web_sm`

## Installation

### Local Development

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

4. Run the two services in separate terminals:
   ```bash
   # Transcription Service
   cd services/transcription && uvicorn main:app --host 0.0.0.0 --port 8000

   # Text Processing Service
   cd services/text_processing && uvicorn main:app --host 0.0.0.0 --port 8001
   ```

### Docker

1. Build the Docker image:
   ```bash
   cd Data-Pipeline
   docker build -t audioseek .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 -p 8001:8001 -e OPENAI_API_KEY=your_openai_api_key audioseek
   ```

   The container will start both services on ports 8000 and 8001.

## Usage

### Process an Audiobook

First, transcribe the audio:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@your_audiobook.mp3"
```

Then, process the transcript through the text processing service:

```bash
curl -X POST "http://localhost:8001/process-full" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "path/to/transcript.txt"}'
```

This will:
1. Chunk the text with entity extraction
2. Generate embeddings
3. Store everything in vector DB and metadata DB

### Ask Questions

Query the text processing service with natural language questions:

```bash
curl -X POST "http://localhost:8001/qa/ask" \
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

#### Text Processing Service (8001)
- `POST /chunk`: Process transcript, returns chunks, chapters, entities
- `POST /embed`: Generate embeddings for text list
- `POST /vector-db/add`: Store embeddings with metadata
- `POST /vector-db/search`: Semantic search by embedding
- `POST /vector-db/query`: Query vector DB with text
- `POST /process-full`: Full pipeline (chunk + embed + vector DB)
- `POST /metadata/*`: Metadata database operations
- `POST /qa/ask`: Answer questions with citations

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
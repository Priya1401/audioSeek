# AudioSeek - Audiobook Q&A System

Production-grade MLOps pipeline for converting audiobooks into searchable Q&A systems with automated validation, bias detection, and experiment tracking.

## Overview

AudioSeek processes audiobooks through transcription, intelligent chunking, vector embeddings, and LLM-powered Q&A. Built with comprehensive validation, multi-model comparison, and bias detection across data slices.

## Features

### Core Pipeline
- **Multi-Model Transcription**: Faster-Whisper with cross-validation against OpenAI Whisper and Wav2Vec2
- **Intelligent Chunking**: Semantic segmentation with spaCy transformer NER (en_core_web_trf)
- **Vector Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) with matched tokenization
- **Dual Storage**: Local FAISS + GCP Vertex AI Vector Search with auto-sync
- **Multi-Book Support**: Isolated indexes per book, automatic chapter detection

### MLOps
- **Validation**: Automated quality checks for transcription, chunking, embeddings, and Q&A
- **Bias Detection**: Statistical analysis across chapters and query types
- **Experiment Tracking**: MLflow for configurations and performance metrics
- **Orchestration**: Airflow DAGs with email alerts and anomaly detection

## Architecture

**Pipeline Flow:**
```
Audio Files
    ↓
Transcription (Faster-Whisper) → Validation (WER < 30%, ROUGE-L > 70%)
    ↓
Cross-Model Comparison (3 models) → Best model selection
    ↓
Chunking (512 tokens, NER) → Validation (consistency, entity coverage)
    ↓
Embeddings (384-dim) → Validation (similarity, diversity, separation)
    ↓
Vector DB (FAISS + GCP) → Upload verification
    ↓
Q&A System (Gemini) → Validation (ROUGE-L, citations, time)
    ↓
Bias Detection → Performance analysis across slices
```

**Components:**
- **Airflow**: Two DAGs (processing, validation)
- **Services**: Transcription (8000), Text Processing (8001)
- **Storage**: FAISS, GCP Storage, SQLite, MLflow
- **APIs**: FastAPI with RESTful endpoints

## Project Structure

```
Data-Pipeline/
├── dags/
│   ├── audio_pipeline_dag.py          # Main processing (transcribe → chunk → embed → store)
│   └── validation_pipeline_dag.py     # Validation (embeddings, Q&A, bias detection)
│
├── services/
│   ├── transcription/                 # Transcription microservice
│   │   ├── main.py                    # FastAPI app
│   │   └── transcription_service.py   # Faster-Whisper logic
│   │
│   └── text_processing/               # Text processing microservice
│       ├── main.py                    # FastAPI app with logging
│       ├── controllers.py             # API endpoints
│       ├── services.py                # Business logic (chunking, embedding, QA)
│       ├── models.py                  # Pydantic request/response models
│       ├── utils.py                   # Chunking, NER, chapter detection
│       ├── config.py                  # Environment configuration
│       ├── faiss_vector_db.py         # Local FAISS implementation
│       ├── gcp_vector_db.py           # GCP Vertex AI implementation
│       └── vector_db_interface.py     # Abstract interface
│
├── scripts/
│   ├── transcription/                 # Batch transcription scripts
│   │   ├── transcription_tasks.py     # Single chapter transcription
│   │   ├── extraction_tasks.py        # Metadata extraction
│   │   └── summary.py                 # Report generation
│   │
│   └── validation/
│       ├── model_validation/          # Reference transcription validation
│       │   ├── transcribe_reference.py
│       │   └── validate_transcription.py
│       │
│       ├── cross_model_evaluation/    # Multi-model comparison
│       │   ├── cross_model_sample_openaiwhisper.py
│       │   ├── cross_model_sample_wav2vec.py
│       │   └── validate_transcription.py
│       │
│       └── QA/                        # Q&A and embedding validation
│           ├── qa_validation.py       # Q&A performance testing
│           ├── embedding_validation.py # Embedding quality checks
│           └── bias_detection.py      # Bias across data slices
│
├── data/
│   ├── raw/                           # Input audio files
│   ├── transcription_results/         # Transcribed text by book
│   ├── validation/                    # Validation reports
│   │   ├── qa_validation_{book}.json
│   │   ├── embedding_validation_{book}.json
│   │   ├── bias_detection_{book}.json
│   │   └── comprehensive_validation_{book}.json
│   └── model_registry/                # Config versions
│
├── faiss_store/                       # Local FAISS indexes by book_id
├── tests/                             # Unit and integration tests
├── docker-compose.yaml                # Multi-service orchestration
├── Dockerfile / Dockerfile.airflow    # Container definitions
└── requirements-services.txt          # Python dependencies
```

## Installation

### 1. Setup

```bash
git clone <repo>
cd Data-Pipeline

# Configure environment
cp .env.example .env
# Edit: GCP_PROJECT_ID, GEMINI_API_KEY, GCS_BUCKET_NAME, SMTP credentials

# Add GCP credentials
cp /path/to/service-account-key.json gcp-credentials.json
```

### 2. Start Services

```bash
docker-compose up -d

# Verify
docker-compose ps
docker-compose logs -f
```

### 3. Access Points

- Airflow UI: http://localhost:8080 (airflow2/airflow2)
- Transcription API: http://localhost:8000/docs
- Text Processing API: http://localhost:8001/docs
- MLflow: http://localhost:5000 (if configured)

## Usage

### 1. Process an Audiobook

**Trigger in Airflow UI:** `audio_processing_pipeline`

```json
{
  "transcription_inputdir": "data/raw/romeo_and_juliet",
  "transcription_type": "audiobook",
  "chunk_target_tokens": 512,
  "chunk_overlap_tokens": 100
}
```

**Pipeline stages:**
1. Reference transcription validation (WER/ROUGE-L)
2. Audio metadata extraction
3. Parallel transcription (4 workers max)
4. Cross-model validation (3 models)
5. Chunking with entity extraction
6. Embedding generation
7. Vector DB population (FAISS + GCP)
8. GCP upload verification

### 2. Validate Results

**Trigger in Airflow UI:** `model_validation_pipeline`

```json
{
  "book_id": "romeo_and_juliet",
  "min_rouge_score": 0.4,
  "min_citation_count": 1
}
```

**Validation tasks (run in parallel):**
- Embedding quality (dimension, similarity, diversity)
- Q&A performance (ROUGE-L, citations, response time)
- Bias detection (chapter performance variance)

**Outputs:** `data/validation/comprehensive_validation_{book_id}.json`

### 3. Query via API

```bash
# Ask a question
curl -X POST http://localhost:8001/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "romeo_and_juliet",
    "query": "What happened in chapter 1?",
    "top_k": 5
  }'

# Get vector DB stats
curl http://localhost:8001/vector-db/stats?book_id=romeo_and_juliet

# Process new book via API (alternative to Airflow)
curl -X POST http://localhost:8001/process-full \
  -d '{
    "book_id": "new_book",
    "folder_path": "/app/raw_data/new_book",
    "add_to_vector_db": true
  }'
```

## API Endpoints

### Text Processing Service (Port 8001)

| Endpoint | Description |
|----------|-------------|
| `POST /chunk` | Chunk transcripts with NER |
| `POST /embed` | Generate embeddings |
| `POST /process-full` | Full pipeline: chunk + embed + vector DB |
| `POST /vector-db/add-from-files` | Load and index from files |
| `POST /vector-db/query` | Semantic search with text query |
| `GET /vector-db/stats` | Database statistics |
| `POST /qa/ask` | Ask questions about audiobook |

## Configuration

### Environment Variables (.env)

```bash
# GCP
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_NAME=your-bucket
GEMINI_API_KEY=your-key
VECTOR_DB_TYPE=gcp  # or 'faiss' for local only

# Airflow Limits (adjust for available RAM)
AIRFLOW__CORE__PARALLELISM=4
AIRFLOW__CORE__DAG_CONCURRENCY=4

# Alerts
ALERT_EMAILS=admin@example.com
```

### Processing Parameters

```python
# Transcription
model: 'base'           # Faster-Whisper model size
beam_size: 5            # Accuracy vs speed
compute_type: 'float32' # Precision

# Chunking
target_tokens: 512      # Chunk size
overlap_tokens: 100     # Context overlap

# Validation Thresholds
WER: < 30%
ROUGE-L: > 70% (transcription), > 40% (Q&A)
Bias threshold: 20% variation
```

## Validation Metrics

**Transcription:**
- WER < 30%, ROUGE-L > 70%
- Cross-model comparison (Faster-Whisper, OpenAI Whisper, Wav2Vec2)

**Embeddings:**
- Dimension: 384 (all-MiniLM-L6-v2)
- Pairwise similarity: 0.1 - 0.7
- Diversity score > 0.3
- Chapter separation > 0.1

**Q&A:**
- ROUGE-L > 40% vs expected answers
- Citations ≥ 1 per answer
- Response time < 15s
- Pass rate ≥ 70%

**Bias Detection:**
- Citation count variance across chapters
- Query complexity performance
- Automated recommendations

## Data Management

### DVC Workflow

```bash
# Track processed data
dvc add data/transcription_results/
dvc add faiss_store/

# Push to remote
dvc push

# Commit to git
git add data/*.dvc .gitignore
git commit -m "Add processed audiobook data"
git push
```

### Storage Locations

- **FAISS**: `faiss_store/{book_id}/`
- **GCP**: `gs://{bucket}/vector-db/{book_id}/`
- **Metadata**: `audiobook_metadata.db` (SQLite)
- **Validation**: `data/validation/`

## Development

### Local Setup

```bash
pip install -r requirements-services.txt
python -m spacy download en_core_web_trf

# Run service locally
cd services/text_processing
uvicorn main:app --reload --port 8001
```

### Testing

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### Adding Test Cases

Create `data/validation/test_cases/{book_id}.json`:
```json
{
  "book_id": "your_book",
  "test_cases": [
    {
      "query": "What is the main theme?",
      "expected_answer": "Expected answer text",
      "query_type": "analytical"
    }
  ]
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM errors | Reduce `AIRFLOW__CORE__PARALLELISM` to 2 |
| Disk full | `docker system prune -a --volumes -f` |
| Null chapter_id | Check filename pattern: `book_chapter_XX.txt` |
| Validation fails | Review reports in `data/validation/` |

## Monitoring

```bash
# Health checks
curl http://localhost:8001/health

# Logs
docker-compose logs -f transcription-textprocessing
docker-compose logs -f airflow-scheduler

# Validation reports
ls data/validation/
```

## Tech Stack

- **ML Models**: Faster-Whisper, Sentence-Transformers, spaCy, Gemini Flash
- **Infrastructure**: Airflow 2.7+, FastAPI, Docker, FAISS, GCP
- **MLOps**: MLflow, DVC, automated validation

---

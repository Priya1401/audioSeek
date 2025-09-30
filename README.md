# audioSeek — Timestamp-Grounded Q&A for Audiobooks & Podcasts

**Ask long-form audio questions and jump straight to the answer.**  
audioSeek transcribes episodes/books, chunks text with timestamps, embeds into a vector database, and uses Retrieval-Augmented Generation (RAG) to return **cited answers with clickable timestamps**. The system supports **spoiler-safe** answers (respecting listener progress), **response caching** for speed/cost, and a **request queue** to grow the library organically.

---

## Problem · Why audioSeek
- **Passive, hard-to-navigate audio**: It’s tedious to skim or revisit long chapters/episodes.
- **No timestamp-grounded answers**: Users want precise, clickable points in audio.
- **Spoiler safety**: Answers should reflect only what the listener has heard so far.
- **Reproducibility & cost**: Transcripts, embeddings, and indices need versioning; repeated work should be cached.
- **Scalable ingestion**: Avoid manual backlogs; process once, share for all.

---

## Features
- **Instant, contextual Q&A** over indexed audiobooks/podcasts.
- **Timestamp-cited answers** with click-to-play.
- **User-driven library growth**: upload/request titles; processed once, searchable for all.
- **Semantic search & context retrieval** to surface the most relevant sections.
- **Response caching** for faster repeat/similar queries and lower compute cost.
- **Scalable serving path** with request routing and load balancing.
- **Centralized data layer** separating audio storage, vector embeddings, and cached responses.
- **Clean API endpoints** for ingest, search, and Q&A.

---

## Architecture (High-Level)

**Query Interface (Frontend)**  
- Two workflows: (1) Ask questions on existing titles; (2) Upload/request new titles.

**API Gateway**  
- **Request Entry**: routes requests to ML services.  
- **Load Balancer**: handles concurrent users.  
- **Cache Check**: returns cached answers when available before invoking RAG.

**Data Layer**  
- **Audio Database/Storage**: temporary store for uploaded audio.  
- **Vector Database**: embeddings for chunks.  
- **Context Retrieval Module**: builds answer context from top-K vector hits.  
- **Response Cache Module**: stores recent answers/snippets for speed & cost control.

**ML Layer**  
- **Transcription Service**: ASR to produce text.  
- **Text Chunking**: sentence-aware splits with **start/end timestamps**.  
- **Embedding Service**: generates vectors and upserts to Vector DB.  
- **Process Query**: normalizes query intent; **Vector Search** retrieves chunks.  
- **LLM Service**: composes context → **must-cite** answers (chapter/timestamps) → caches response.

> This design minimizes redundant ASR/embedding work and scales cost-effectively as the library grows.

---

## Installation

### Prerequisites
- Python 3.10+

### Setup
```bash
git clone https://github.com/Priya1401/audioSeek.git
cd audioseek
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt

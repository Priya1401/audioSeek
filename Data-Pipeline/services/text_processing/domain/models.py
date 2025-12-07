from typing import List, Dict, Optional, Any

from pydantic import BaseModel


# -------------------------
# Chunking
# -------------------------
class ChunkingRequest(BaseModel):
    book_id: Optional[str] = None  # Auto-extract if not provided
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 300
    overlap_tokens: int = 50
    output_file: Optional[str] = None


class ChunkResponse(BaseModel):
    book_id: str  # Always return the book_id used
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    processed_files: List[str]
    output_file: Optional[str] = None


# -------------------------
# Embeddings
# -------------------------
class EmbeddingRequest(BaseModel):
    book_id: Optional[str] = None  # Auto-extract if chunks_file provided
    texts: Optional[List[str]] = None
    chunks_file: Optional[str] = None
    output_file: Optional[str] = None


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    count: int
    output_file: Optional[str] = None


# -------------------------
# Combined Pipeline
# -------------------------
class CombinedRequest(BaseModel):
    book_id: Optional[str] = None  # Auto-extract if not provided
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 300
    overlap_tokens: int = 50
    chunks_output_file: Optional[str] = None
    embeddings_output_file: Optional[str] = None


class CombinedResponse(BaseModel):
    book_id: str  # Add this field
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    embeddings: List[List[float]]
    processed_files: List[str]
    chunks_output_file: Optional[str] = None
    embeddings_output_file: Optional[str] = None  # Add this field


# -------------------------
# Full Pipeline
# -------------------------
class FullPipelineRequest(BaseModel):
    book_id: Optional[str] = None  # Auto-extract if not provided
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 300
    overlap_tokens: int = 50
    add_to_vector_db: bool = True


class FullPipelineResponse(BaseModel):
    book_id: str  # Add this field
    chunks_count: int
    chapters_count: int
    entities_count: int
    embeddings_count: int
    vector_db_added: bool
    files_processed: int
    message: str


# -------------------------
# Vector DB
# -------------------------
class AddDocumentsRequest(BaseModel):
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]
    book_id: Optional[str] = None  # Changed from "default"


class AddDocumentsResponse(BaseModel):
    message: str
    count: int


class AddFromFilesRequest(BaseModel):
    chunks_file: str
    embeddings_file: str
    book_id: Optional[str] = None  # Changed from "default"


class AddFromFilesResponse(BaseModel):
    message: str
    chunks_count: int
    embeddings_count: int


class SearchRequest(BaseModel):
    query_embedding: List[float]
    top_k: int = 10
    book_id: Optional[str] = None
    chapter: Optional[int] = None
    time_seconds: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int


# -------------------------
# QA
# -------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    book_id: Optional[str] = None
    session_id: Optional[str] = None
    until_chapter: Optional[int] = None
    until_time_seconds: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    session_id: Optional[str] = None
    audio_references: List[Dict[str, Any]] = []

# -------------------------
# Transcription
# -------------------------
class TranscriptionRequest(BaseModel):
    folder_path: str
    content_type: str = "audiobook"
    model_size: str = "base"
    beam_size: int = 5
    compute_type: str = "float32"

class AudioProcessRequest(BaseModel):
    folder_path: str
    book_name: str
    content_type: str = "audiobook"
    target_tokens: int = 512
    overlap_tokens: int = 50
    model_size: str = "base"
    beam_size: int = 5
    compute_type: str = "float32"
    user_email: Optional[str] = "anonymous"


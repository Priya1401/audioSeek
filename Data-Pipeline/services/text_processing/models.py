from typing import List, Dict, Optional, Any
from pydantic import BaseModel


# -------------------------
# Chunking
# -------------------------
class ChunkingRequest(BaseModel):
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 300
    overlap_tokens: int = 50
    output_file: Optional[str] = None


class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    processed_files: List[str]
    output_file: Optional[str] = None


# -------------------------
# Embeddings
# -------------------------
class EmbeddingRequest(BaseModel):
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
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 300
    overlap_tokens: int = 50
    chunks_output_file: Optional[str] = None
    embeddings_output_file: Optional[str] = None
    book_id: Optional[str] = "default"


class CombinedResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    embeddings: List[List[float]]
    processed_files: List[str]
    chunks_output_file: Optional[str] = None


# -------------------------
# Full Pipeline
# -------------------------
class FullPipelineRequest(BaseModel):
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 300
    overlap_tokens: int = 50
    add_to_vector_db: bool = True
    book_id: Optional[str] = "default"


class FullPipelineResponse(BaseModel):
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
    book_id: Optional[str] = "default"


class AddDocumentsResponse(BaseModel):
    message: str
    count: int


class AddFromFilesRequest(BaseModel):
    chunks_file: str
    embeddings_file: str
    book_id: Optional[str] = "default"


class AddFromFilesResponse(BaseModel):
    message: str
    chunks_count: int
    embeddings_count: int


class SearchRequest(BaseModel):
    query_embedding: List[float]
    top_k: int = 5
    book_id: Optional[str] = "default"


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int


# -------------------------
# QA
# -------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    book_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]

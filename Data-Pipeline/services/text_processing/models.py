from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ============= CHUNKING MODELS =============
class ChunkingRequest(BaseModel):
    file_path: str
    target_tokens: int = 512
    overlap_tokens: int = 100
    output_file: Optional[str] = None

class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    output_file: Optional[str] = None

# ============= EMBEDDING MODELS =============
class EmbeddingRequest(BaseModel):
    texts: Optional[List[str]] = None
    chunks_file: Optional[str] = None
    output_file: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    count: int
    output_file: Optional[str] = None

# ============= VECTOR DB MODELS =============
class AddDocumentsRequest(BaseModel):
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]

class AddDocumentsResponse(BaseModel):
    message: str
    count: int

class SearchRequest(BaseModel):
    query_embedding: List[float]
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int

class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 5

# ============= COMBINED MODELS =============
class CombinedRequest(BaseModel):
    file_path: str
    target_tokens: int = 512
    overlap_tokens: int = 100
    chunks_output_file: Optional[str] = None
    embeddings_output_file: Optional[str] = None

class CombinedResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    embeddings: List[List[float]]
    chunks_output_file: Optional[str] = None
    embeddings_output_file: Optional[str] = None

class FullPipelineRequest(BaseModel):
    file_path: str
    target_tokens: int = 512
    overlap_tokens: int = 100
    add_to_vector_db: bool = True

class FullPipelineResponse(BaseModel):
    chunks_count: int
    chapters_count: int
    entities_count: int
    embeddings_count: int
    vector_db_added: bool
    message: str

# ============= METADATA DB MODELS =============
class AudiobookCreate(BaseModel):
    title: str
    author: Optional[str] = None
    duration: Optional[float] = None

class ChapterCreate(BaseModel):
    audiobook_id: int
    title: str
    start_time: float
    end_time: float
    summary: Optional[str] = None

class ChunkCreate(BaseModel):
    audiobook_id: int
    chapter_id: Optional[int] = None
    start_time: float
    end_time: float
    text: str
    token_count: int
    embedding_id: int

class EntityCreate(BaseModel):
    name: str
    type: str
    audiobook_id: int

class EntityMentionCreate(BaseModel):
    entity_id: int
    chunk_id: int
    start_pos: int
    end_pos: int

# ============= QA MODELS =============
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
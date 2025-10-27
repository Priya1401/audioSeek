from pydantic import BaseModel, model_validator
from typing import List, Dict, Any, Optional

# ============= CHUNKING MODELS =============
class ChunkingRequest(BaseModel):
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 512
    overlap_tokens: int = 100
    output_file: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_at_least_one_path(self):
        if not any([self.file_path, self.file_paths, self.folder_path]):
            raise ValueError("Must provide file_path, file_paths, or folder_path")
        return self

class AddFromFilesRequest(BaseModel):
    chunks_file: str
    embeddings_file: str

class AddFromFilesResponse(BaseModel):
    message: str
    chunks_count: int
    embeddings_count: int

class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    processed_files: List[str]
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
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 512
    overlap_tokens: int = 100
    chunks_output_file: Optional[str] = None
    embeddings_output_file: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_at_least_one_path(self):
        if not any([self.file_path, self.file_paths, self.folder_path]):
            raise ValueError("Must provide file_path, file_paths, or folder_path")
        return self

class CombinedResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    embeddings: List[List[float]]
    processed_files: List[str]
    chunks_output_file: Optional[str] = None
    embeddings_output_file: Optional[str] = None

class FullPipelineRequest(BaseModel):
    file_path: Optional[str] = None
    file_paths: Optional[List[str]] = None
    folder_path: Optional[str] = None
    target_tokens: int = 512
    overlap_tokens: int = 100
    add_to_vector_db: bool = True
    
    @model_validator(mode='after')
    def validate_at_least_one_path(self):
        if not any([self.file_path, self.file_paths, self.folder_path]):
            raise ValueError("Must provide file_path, file_paths, or folder_path")
        return self

class FullPipelineResponse(BaseModel):
    chunks_count: int
    chapters_count: int
    entities_count: int
    embeddings_count: int
    vector_db_added: bool
    files_processed: int
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
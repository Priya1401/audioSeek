from fastapi import APIRouter
from typing import Optional
from models import (
    ChunkingRequest,
    ChunkResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    CombinedRequest,
    CombinedResponse,
    AddDocumentsRequest,
    AddDocumentsResponse,
    SearchRequest,
    SearchResponse,
    QueryRequest,
    FullPipelineRequest,
    FullPipelineResponse,
    AudiobookCreate,
    ChapterCreate,
    ChunkCreate,
    EntityCreate,
    EntityMentionCreate,
    QueryResponse
)
from services import (
    ChunkingService,
    EmbeddingService,
    VectorDBService,
    PipelineService,
    MetadataDBService,
    QAService
)
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

router = APIRouter()

# Initialize services
metadata_db = MetadataDBService()
vector_db = VectorDBService()
qa_service = QAService(metadata_db, vector_db, embedding_model)

# ============= CHUNKING ENDPOINTS =============
@router.post("/chunk", response_model=ChunkResponse, tags=["Chunking"])
async def chunk_transcript(request: ChunkingRequest):
    """Chunk a transcript file into segments with chapter detection and entity extraction"""
    return ChunkingService.chunk_transcript(request)

# ============= EMBEDDING ENDPOINTS =============
@router.post("/embed", response_model=EmbeddingResponse, tags=["Embedding"])
async def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for text chunks"""
    return EmbeddingService.generate_embeddings(request)

# ============= VECTOR DB ENDPOINTS =============
@router.post("/vector-db/add", response_model=AddDocumentsResponse, tags=["Vector DB"])
async def add_documents(request: AddDocumentsRequest):
    """Add documents with embeddings to the vector database"""
    return VectorDBService.add_documents(request)

@router.post("/vector-db/search", response_model=SearchResponse, tags=["Vector DB"])
async def search_vectors(request: SearchRequest):
    """Search for similar vectors in the database using an embedding"""
    return VectorDBService.search(request)

@router.post("/vector-db/query", response_model=SearchResponse, tags=["Vector DB"])
async def query_text(request: QueryRequest):
    """Query the vector database with text (embedding generated automatically)"""
    return VectorDBService.query_text(request)

@router.get("/vector-db/stats", tags=["Vector DB"])
async def get_vector_db_stats():
    """Get vector database statistics"""
    return VectorDBService.get_stats()

# ============= PIPELINE ENDPOINTS =============
@router.post("/process", response_model=CombinedResponse, tags=["Pipeline"])
async def process_combined_pipeline(request: CombinedRequest):
    """Run chunking and embedding pipeline"""
    return PipelineService.process_combined_pipeline(request)

@router.post("/process-full", response_model=FullPipelineResponse, tags=["Pipeline"])
async def process_full_pipeline(request: FullPipelineRequest):
    """Run the complete pipeline: chunk, embed, and add to vector DB"""
    return PipelineService.process_full_pipeline(request)

# ============= METADATA DB ENDPOINTS =============
@router.post("/metadata/audiobooks", tags=["Metadata DB"])
async def create_audiobook(audiobook: AudiobookCreate):
    """Create a new audiobook"""
    return metadata_db.create_audiobook(audiobook)

@router.post("/metadata/chapters", tags=["Metadata DB"])
async def create_chapter(chapter: ChapterCreate):
    """Create a new chapter"""
    return metadata_db.create_chapter(chapter)

@router.post("/metadata/chunks", tags=["Metadata DB"])
async def create_chunk(chunk: ChunkCreate):
    """Create a new chunk"""
    return metadata_db.create_chunk(chunk)

@router.post("/metadata/entities", tags=["Metadata DB"])
async def create_entity(entity: EntityCreate):
    """Create a new entity"""
    return metadata_db.create_entity(entity)

@router.post("/metadata/entity_mentions", tags=["Metadata DB"])
async def create_entity_mention(mention: EntityMentionCreate):
    """Create a new entity mention"""
    return metadata_db.create_entity_mention(mention)

@router.get("/metadata/audiobooks/{audiobook_id}/chapters", tags=["Metadata DB"])
async def get_chapters(audiobook_id: int):
    """Get chapters for an audiobook"""
    return metadata_db.get_chapters(audiobook_id)

@router.get("/metadata/chunks", tags=["Metadata DB"])
async def get_chunks(audiobook_id: int, chapter_id: Optional[int] = None):
    """Get chunks for an audiobook, optionally filtered by chapter"""
    return metadata_db.get_chunks(audiobook_id, chapter_id)

@router.get("/metadata/entities/{audiobook_id}", tags=["Metadata DB"])
async def get_entities(audiobook_id: int):
    """Get entities for an audiobook"""
    return metadata_db.get_entities(audiobook_id)

# ============= QA ENDPOINTS =============
@router.post("/qa/ask", response_model=QueryResponse, tags=["QA"])
async def ask_question(request: QueryRequest):
    """Answer questions about the audiobook"""
    return qa_service.ask_question(request)

# ============= ROOT ENDPOINT =============
@router.get("/", tags=["Root"])
async def root():
    return {
        "service": "Text Processing Service",
        "status": "healthy",
        "endpoints": {
            "/chunk": "Chunk transcripts",
            "/embed": "Generate embeddings",
            "/vector-db/add": "Add documents to vector DB",
            "/vector-db/search": "Search vector DB with embedding",
            "/vector-db/query": "Query vector DB with text",
            "/vector-db/stats": "Get vector DB statistics",
            "/process": "Combined pipeline (chunk + embed)",
            "/process-full": "Full pipeline (chunk + embed + vector DB)",
            "/metadata/*": "Metadata database operations",
            "/qa/ask": "Question answering"
        }
    }
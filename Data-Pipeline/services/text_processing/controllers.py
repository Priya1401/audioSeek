from fastapi import APIRouter
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
    FullPipelineResponse
)
from services import (
    ChunkingService, 
    EmbeddingService, 
    VectorDBService,
    PipelineService
)

router = APIRouter()

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
            "/process-full": "Full pipeline (chunk + embed + vector DB)"
        }
    }
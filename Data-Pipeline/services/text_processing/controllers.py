from fastapi import APIRouter, HTTPException

from models import (
    ChunkingRequest,
    CombinedRequest,
    FullPipelineRequest,
    EmbeddingRequest,
    QueryRequest,
    AddDocumentsRequest,
    SearchRequest,
)
from services import (
    ChunkingService,
    EmbeddingService,
    PipelineService,
    QAService,
    MetadataDBService,
    VectorDBService
)

router = APIRouter()

# Global metadata DB instance
metadata_db = MetadataDBService()

# QA service (uses dynamic vector DB inside)
qa_service = QAService(metadata_db)


# --------------------------------------------------------
# CHUNKING ENDPOINT
# --------------------------------------------------------
@router.post("/chunk")
async def chunk_transcript(request: ChunkingRequest):
    return ChunkingService.chunk_transcript(request)


# --------------------------------------------------------
# EMBEDDING ENDPOINT
# --------------------------------------------------------
@router.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    return EmbeddingService.generate_embeddings(request)


# --------------------------------------------------------
# COMBINED PIPELINE (chunk + embed)
# --------------------------------------------------------
@router.post("/process")
async def process_combined(request: CombinedRequest):
    return PipelineService.process_combined_pipeline(request)


# --------------------------------------------------------
# FULL PIPELINE (chunk + embed + metadata + FAISS)
# --------------------------------------------------------
@router.post("/process-full")
async def process_full(request: FullPipelineRequest):
    return PipelineService.process_full_pipeline(request)


# --------------------------------------------------------
# VECTOR DB — ADD DOCUMENTS
# --------------------------------------------------------
@router.post("/vector-db/add-documents")
async def add_documents(request: AddDocumentsRequest):
    return VectorDBService.add_documents(request)


# --------------------------------------------------------
# VECTOR DB — SEARCH
# --------------------------------------------------------
@router.post("/vector-db/search")
async def vector_search(request: SearchRequest):
    return VectorDBService.search(request)


# --------------------------------------------------------
# VECTOR DB — QUERY (text → embedding → search)
# --------------------------------------------------------
@router.post("/vector-db/query")
async def vector_query(request: QueryRequest):
    return VectorDBService.query_text(request)


# --------------------------------------------------------
# VECTOR DB — STATS PER BOOK
# --------------------------------------------------------
@router.get("/vector-db/stats")
async def vector_stats(book_id: str = "default"):
    return VectorDBService.get_stats(book_id)


# --------------------------------------------------------
# QA ENDPOINT
# --------------------------------------------------------
@router.post("/qa/ask")
async def qa_ask(request: QueryRequest):
    return qa_service.ask_question(request)

from fastapi import APIRouter, HTTPException
import mlflow
from config_mlflow import MLFLOW_EXPERIMENT_NAME
import time

from models import (
    ChunkingRequest,
    CombinedRequest,
    FullPipelineRequest,
    EmbeddingRequest,
    QueryRequest,
    AddDocumentsRequest,
    AddFromFilesRequest,
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


@router.get("/books")
async def list_books():
    """List all available audiobooks"""
    return metadata_db.get_all_audiobooks()


# --------------------------------------------------------
# CHUNKING ENDPOINT
# --------------------------------------------------------
@router.post("/chunk")
async def chunk_transcript(request: ChunkingRequest):
    # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # with mlflow.start_run(run_name=f"chunk_{request.book_id or 'default'}"):
    #     mlflow.log_param("book_id", request.book_id or "default")
    #     mlflow.log_param("target_tokens", request.target_tokens)
    #     mlflow.log_param("overlap_tokens", request.overlap_tokens)

        start = time.time()
        resp = ChunkingService.chunk_transcript(request)
        duration = time.time() - start

        # mlflow.log_metric("chunks_count", len(resp.chunks))
        # mlflow.log_metric("chapters_count", len(resp.chapters))
        # mlflow.log_metric("entities_count", len(resp.entities))
        # mlflow.log_metric("chunk_endpoint_time_sec", duration)

        return resp


# --------------------------------------------------------
# EMBEDDING ENDPOINT
# --------------------------------------------------------
@router.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # with mlflow.start_run(run_name="generate_embeddings"):
        source_type = "chunks_file" if request.chunks_file else "raw_texts"
        # mlflow.log_param("source_type", source_type)
        if request.chunks_file:
            pass
            # mlflow.log_param("chunks_file", request.chunks_file)

        start = time.time()
        resp = EmbeddingService.generate_embeddings(request)
        duration = time.time() - start

        # mlflow.log_metric("embedding_count", resp.count)
        # mlflow.log_metric("embed_endpoint_time_sec", duration)

        return resp


# --------------------------------------------------------
# COMBINED PIPELINE (chunk + embed)
# (no MLflow here; full tracking is in /process-full)
# --------------------------------------------------------
@router.post("/process")
async def process_combined(request: CombinedRequest):
    return PipelineService.process_combined_pipeline(request)


# --------------------------------------------------------
# FULL PIPELINE (chunk + embed + metadata + FAISS)
# MLflow logging is implemented inside PipelineService.process_full_pipeline
# --------------------------------------------------------
@router.post("/process-full")
async def process_full(request: FullPipelineRequest):
    return PipelineService.process_full_pipeline(request)


# --------------------------------------------------------
# VECTOR DB — ADD DOCUMENTS
# --------------------------------------------------------
@router.post("/vector-db/add-documents")
async def add_documents(request: AddDocumentsRequest):
    # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # with mlflow.start_run(run_name=f"vector_add_{request.book_id or 'default'}"):
    #     mlflow.log_param("book_id", request.book_id or "default")
    #     mlflow.log_metric("docs_to_add", len(request.embeddings))

        start = time.time()
        resp = VectorDBService.add_documents(request)
        duration = time.time() - start

        # mlflow.log_metric("docs_added", resp.count)
        # mlflow.log_metric("vector_add_time_sec", duration)

        return resp


# --------------------------------------------------------
# VECTOR DB — SEARCH
# --------------------------------------------------------
@router.post("/vector-db/search")
async def vector_search(request: SearchRequest):
    # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # with mlflow.start_run(run_name=f"vector_search_{request.book_id or 'default'}"):
    #     mlflow.log_param("book_id", request.book_id or "default")
    #     mlflow.log_param("top_k", request.top_k)
    #     mlflow.log_metric("query_embedding_dim", len(request.query_embedding))

        start = time.time()
        resp = VectorDBService.search(request)
        duration = time.time() - start

        # mlflow.log_metric("results_count", resp.count)
        # mlflow.log_metric("vector_search_time_sec", duration)

        return resp


# --------------------------------------------------------
# VECTOR DB — QUERY (text → embedding → search)
# --------------------------------------------------------
@router.post("/vector-db/query")
async def vector_query(request: QueryRequest):
    # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # with mlflow.start_run(run_name=f"vector_query_{request.book_id or 'default'}"):
    #     mlflow.log_param("book_id", request.book_id or "default")
    #     mlflow.log_param("top_k", request.top_k)
    #     # Avoid logging massive queries as params
    #     if request.query:
    #         pass
    #         # mlflow.log_param("query_preview", request.query[:200])

        start = time.time()
        resp = VectorDBService.query_text(request)
        duration = time.time() - start

        # mlflow.log_metric("results_count", resp.count)
        # mlflow.log_metric("vector_query_time_sec", duration)

        return resp


# --------------------------------------------------------
# VECTOR DB — STATS PER BOOK
# --------------------------------------------------------
@router.get("/vector-db/stats")
async def vector_stats(book_id: str = "default"):
    return VectorDBService.get_stats(book_id)


# --------------------------------------------------------
# QA ENDPOINT
# MLflow logging is implemented inside QAService.ask_question
# --------------------------------------------------------
@router.post("/qa/ask")
async def qa_ask(request: QueryRequest):
    return qa_service.ask_question(request)
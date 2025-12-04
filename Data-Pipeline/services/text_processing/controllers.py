from fastapi import APIRouter, HTTPException
import mlflow
from config_mlflow import MLFLOW_EXPERIMENT_NAME
import time
from fastapi import UploadFile, File, Form
import uuid
import zipfile
import shutil
from pathlib import Path

from models import (
    ChunkingRequest,
    CombinedRequest,
    FullPipelineRequest,
    EmbeddingRequest,
    QueryRequest,
    AddDocumentsRequest,
    AddFromFilesRequest,
    SearchRequest,
    TranscriptionRequest,
    AudioProcessRequest
)

from services import (
    ChunkingService,
    EmbeddingService,
    PipelineService,
    QAService,
    MetadataDBService,
    VectorDBService
)

from transcription_service import TranscriptionService

router = APIRouter()

# Global metadata DB instance
metadata_db = MetadataDBService()

# QA service (uses dynamic vector DB inside)
qa_service = QAService(metadata_db)

# NEW transcription service
transcription_service = TranscriptionService()


# --------------------------------------------------------
# TRANSCRIPTION ENDPOINT  (NEW)
# --------------------------------------------------------
@router.post("/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = f"transcribe_{request.book_name}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("folder_path", request.folder_path)
        mlflow.log_param("book_name", request.book_name)

        resp = transcription_service.transcribe_directory(
            folder_path=request.folder_path,
            book_name=request.book_name,
            model_size=request.model_size,
            beam_size=request.beam_size,
            compute_type=request.compute_type
        )

        mlflow.log_metric("transcriptions_count", len(resp["files"]))

        return resp



# --------------------------------------------------------
# AUDIO INGESTION FULL PIPELINE (NEW)
# (transcribe → chunk → embed → vector DB)
# --------------------------------------------------------
@router.post("/process-audio")
async def process_audio_pipeline(request: AudioProcessRequest):
    """
    End-to-end ingest:
    1. Transcribe audio files
    2. Chunk transcripts
    3. Embed chunks
    4. Add to vector DB
    """

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = f"process_audio_{request.book_name}"

    with mlflow.start_run(run_name=run_name):
        # Log pipeline-level params
        mlflow.log_param("folder_path", request.folder_path)
        mlflow.log_param("book_name", request.book_name)
        mlflow.log_param("content_type", request.content_type)
        mlflow.log_param("model_size", request.model_size)
        mlflow.log_param("beam_size", request.beam_size)
        mlflow.log_param("compute_type", request.compute_type)

        # ------------------ STEP 1: TRANSCRIBE ------------------
        start_trans = time.time()
        transcripts = transcription_service.transcribe_directory(
            folder_path=request.folder_path,
            book_name=request.book_name,
            model_size=request.model_size,
            beam_size=request.beam_size,
            compute_type=request.compute_type
        )
        mlflow.log_metric("transcription_time_sec", time.time() - start_trans)
        mlflow.log_metric("transcriptions_count", len(transcripts["files"]))

        # ------------------ STEP 2: CHUNK -----------------------
        chunk_output_file = f"/app/raw_data/transcription_results/{request.book_name}/chunks_output.json"

        chunk_request = ChunkingRequest(
            folder_path=f"/app/raw_data/transcription_results/{request.book_name}",
            book_id=request.book_name,
            target_tokens=request.target_tokens,
            overlap_tokens=request.overlap_tokens,
            output_file=chunk_output_file
        )

        start_chunk = time.time()
        chunk_resp = ChunkingService.chunk_transcript(chunk_request)
        mlflow.log_metric("chunk_time_sec", time.time() - start_chunk)
        mlflow.log_metric("chunks_count", len(chunk_resp.chunks))
        mlflow.log_param("chunk_output_file", chunk_resp.output_file)

        # ------------------ STEP 3: EMBED -----------------------
        embed_output_file = f"/app/raw_data/transcription_results/{request.book_name}/embeddings_output.json"

        embed_request = EmbeddingRequest(
            chunks_file=chunk_resp.output_file,
            book_id=request.book_name,
            output_file=embed_output_file
        )

        start_embed = time.time()
        embed_resp = EmbeddingService.generate_embeddings(embed_request)
        mlflow.log_metric("embedding_time_sec", time.time() - start_embed)
        mlflow.log_metric("embedding_count", embed_resp.count)
        mlflow.log_param("embedding_output_file", embed_resp.output_file)

        # ------------------ STEP 4: VECTOR DB -------------------
        add_request = AddFromFilesRequest(
            chunks_file=chunk_resp.output_file,
            embeddings_file=embed_resp.output_file,
            book_id=request.book_name
        )

        start_db = time.time()
        add_resp = VectorDBService.add_from_files(add_request)
        # mlflow.log_metric("vector_db_write_sec", time.time() - start_db)
        # mlflow.log_metric("vector_count", add_resp.embeddings_count)

        # ------------------ FINAL RESPONSE -----------------------
        return {
            "status": "complete",
            "book_name": request.book_name,
            "transcription": transcripts,
            "chunking": chunk_resp,
            "embedding": embed_resp,
            "vector_db": add_resp
        }



@router.get("/books")
async def list_books():
    """List all available audiobooks"""
    return metadata_db.get_all_audiobooks()


# --------------------------------------------------------
# CHUNKING ENDPOINT
# --------------------------------------------------------
@router.post("/chunk")
async def chunk_transcript(request: ChunkingRequest):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"chunk_{request.book_id or 'default'}"):
        mlflow.log_param("book_id", request.book_id or "default")
        mlflow.log_param("target_tokens", request.target_tokens)
        mlflow.log_param("overlap_tokens", request.overlap_tokens)

        start = time.time()
        resp = ChunkingService.chunk_transcript(request)
        duration = time.time() - start

        mlflow.log_metric("chunks_count", len(resp.chunks))
        mlflow.log_metric("chapters_count", len(resp.chapters))
        mlflow.log_metric("entities_count", len(resp.entities))
        mlflow.log_metric("chunk_endpoint_time_sec", duration)

        return resp


# --------------------------------------------------------
# EMBEDDING ENDPOINT
# --------------------------------------------------------
@router.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="generate_embeddings"):
        source_type = "chunks_file" if request.chunks_file else "raw_texts"
        mlflow.log_param("source_type", source_type)
        if request.chunks_file:
            mlflow.log_param("chunks_file", request.chunks_file)

        start = time.time()
        resp = EmbeddingService.generate_embeddings(request)
        duration = time.time() - start

        mlflow.log_metric("embedding_count", resp.count)
        mlflow.log_metric("embed_endpoint_time_sec", duration)

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
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"vector_add_{request.book_id or 'default'}"):
        mlflow.log_param("book_id", request.book_id or "default")
        mlflow.log_metric("docs_to_add", len(request.embeddings))

        start = time.time()
        resp = VectorDBService.add_documents(request)
        duration = time.time() - start

        mlflow.log_metric("docs_added", resp.count)
        mlflow.log_metric("vector_add_time_sec", duration)

        return resp


# --------------------------------------------------------
# VECTOR DB — SEARCH
# --------------------------------------------------------
@router.post("/vector-db/search")
async def vector_search(request: SearchRequest):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"vector_search_{request.book_id or 'default'}"):
        mlflow.log_param("book_id", request.book_id or "default")
        mlflow.log_param("top_k", request.top_k)
        mlflow.log_metric("query_embedding_dim", len(request.query_embedding))

        start = time.time()
        resp = VectorDBService.search(request)
        duration = time.time() - start

        mlflow.log_metric("results_count", resp.count)
        mlflow.log_metric("vector_search_time_sec", duration)

        return resp


# --------------------------------------------------------
# VECTOR DB — QUERY (text → embedding → search)
# --------------------------------------------------------
@router.post("/vector-db/query")
async def vector_query(request: QueryRequest):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"vector_query_{request.book_id or 'default'}"):
        mlflow.log_param("book_id", request.book_id or "default")
        mlflow.log_param("top_k", request.top_k)
        # Avoid logging massive queries as params
        if request.query:
            mlflow.log_param("query_preview", request.query[:200])

        start = time.time()
        resp = VectorDBService.query_text(request)
        duration = time.time() - start

        mlflow.log_metric("results_count", resp.count)
        mlflow.log_metric("vector_query_time_sec", duration)

        return resp


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


# --------------------------------------------------------
# FILE UPLOAD ENDPOINT
# --------------------------------------------------------
@router.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    book_name: str = Form(...)
):
    book_name = book_name.strip().replace(" ", "_").lower()

    # Unique session folder
    session_id = str(uuid.uuid4())
    upload_dir = Path(f"/app/uploads/{book_name}_{session_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ------------------------------------------------------------
    # ZIP CASE — extract and flatten directory structure
    # ------------------------------------------------------------
    if file.filename.endswith(".zip"):
        extract_dir = upload_dir / book_name
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Remove original zip
        file_path.unlink()

        # ---- FLATTEN ANY NESTED STRUCTURE ----
        # Move all audio files from ANY depth to <extract_dir>
        for audio_file in extract_dir.rglob("*"):
            if audio_file.suffix.lower() in [".mp3", ".wav"]:
                shutil.move(str(audio_file), extract_dir)

        # Clean up leftover directories
        for sub in extract_dir.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)

        return {
            "status": "uploaded_zip",
            "book_name": book_name,
            "folder_path": str(extract_dir)
        }

    # ------------------------------------------------------------
    # SINGLE AUDIO FILE CASE
    # ------------------------------------------------------------
    elif file.filename.endswith((".mp3", ".wav")):
        audio_dir = upload_dir / book_name
        audio_dir.mkdir(exist_ok=True)
        shutil.move(str(file_path), audio_dir)

        return {
            "status": "uploaded_audio",
            "book_name": book_name,
            "folder_path": str(audio_dir)
        }

    else:
        raise HTTPException(400, "Unsupported file format. Upload .mp3, .wav, or .zip")
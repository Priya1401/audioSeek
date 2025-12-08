from fastapi import APIRouter, HTTPException, BackgroundTasks
import mlflow
from core.config_mlflow import MLFLOW_EXPERIMENT_NAME
import time
from fastapi import UploadFile, File, Form
import uuid
import zipfile
import shutil
from pathlib import Path

from domain.models import (
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

from services.nlp.chunking_service import ChunkingService
from services.nlp.embedding_service import EmbeddingService
from services.nlp.pipeline_service import PipelineService
from services.nlp.qa_service import QAService
from services.storage.metadata_db_service import MetadataDBService
from services.storage.vector_db_service import VectorDBService

from services.audio.transcription_service import TranscriptionService
from services.jobs.job_service import job_service
from services.storage.firestore_service import firestore_db
import google.auth
from google.cloud import storage
from google.auth.transport.requests import requests as google_requests
import os

router = APIRouter()

# Global metadata DB instance
metadata_db = MetadataDBService()

# QA service (uses dynamic vector DB inside)
qa_service = QAService(metadata_db)

# NEW transcription service
transcription_service = TranscriptionService()

# Get GCP credentials
credentials, project = google.auth.default()

storage_client = storage.Client(credentials=credentials, project=project)
bucket_name = os.getenv("GCP_BUCKET_NAME", "audioseek-bucket")
bucket = storage_client.bucket(bucket_name)


# --------------------------------------------------------
# TRANSCRIPTION ENDPOINT  (NEW)
# --------------------------------------------------------
@router.post("/transcribe")
def transcribe_audio(request: TranscriptionRequest):
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
def process_audio_pipeline(request: AudioProcessRequest, background_tasks: BackgroundTasks):
    """
    Async end-to-end ingest:
    1. Create Job ID
    2. Return Job ID immediately
    3. Run processing in background
    """
    
    # Create Job
    job = job_service.create_job(request, request.user_email)
    
    # Add to background tasks
    background_tasks.add_task(job_service.process_audio_background, job.job_id, request)
    
    return {
        "status": "submitted",
        "job_id": job.job_id,
        "message": "Processing started in background. Check status at /jobs/{job_id}"
    }


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = firestore_db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/user/{user_email}")
def get_user_jobs(user_email: str):
    return firestore_db.get_user_jobs(user_email)



@router.get("/books")
async def list_books():
    """List all available audiobooks"""
    return metadata_db.get_all_audiobooks()


# --------------------------------------------------------
# CHUNKING ENDPOINT
# --------------------------------------------------------
@router.post("/chunk")
def chunk_transcript(request: ChunkingRequest):
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
def generate_embeddings(request: EmbeddingRequest):
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
def process_combined(request: CombinedRequest):
    return PipelineService.process_combined_pipeline(request)


# --------------------------------------------------------
# FULL PIPELINE (chunk + embed + metadata + FAISS)
# MLflow logging is implemented inside PipelineService.process_full_pipeline
# --------------------------------------------------------
@router.post("/process-full")
def process_full(request: FullPipelineRequest):
    return PipelineService.process_full_pipeline(request)


# --------------------------------------------------------
# VECTOR DB — ADD DOCUMENTS
# --------------------------------------------------------
@router.post("/vector-db/add-documents")
def add_documents(request: AddDocumentsRequest):
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
def vector_search(request: SearchRequest):
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
def vector_query(request: QueryRequest):
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
# ADMIN STATS ENDPOINT
# --------------------------------------------------------
@router.get("/admin/stats")
def get_admin_stats():
    try:
        # Get Metadata DB stats
        db_stats = metadata_db.get_system_stats()
        book_details = metadata_db.get_detailed_book_stats()
        
        # Get Job stats from Firestore
        job_stats = firestore_db.get_all_jobs_stats()
        
        return {
            "database": db_stats,
            "jobs": job_stats,
            "books": book_details,
            "system_status": "healthy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


# --------------------------------------------------------
# BOOK STATUS ENDPOINT
# --------------------------------------------------------
@router.get("/books/{book_id}/status")
def get_book_status(book_id: str):
    try:
        # Check if book exists in Metadata DB
        chapters = metadata_db.get_chapters(book_id)
        chunks = metadata_db.get_chunks(book_id)
        
        if not chapters["chapters"] and not chunks["chunks"]:
             raise HTTPException(status_code=404, detail=f"Book '{book_id}' not found or not processed.")

        return {
            "book_id": book_id,
            "status": "ready",
            "chapters_count": len(chapters["chapters"]),
            "chunks_count": len(chunks["chunks"])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check book status: {str(e)}")


# --------------------------------------------------------
# QA ENDPOINT
# --------------------------------------------------------
@router.post("/qa/ask")
def qa_ask(request: QueryRequest):
    try:
        response = qa_service.ask_question(request)
        
        # Inject Signed URLs if audio references exist
        if response.audio_references:
            import logging
            from google.cloud import storage
            from datetime import timedelta
            import os

            logger = logging.getLogger(__name__)

            try:

                for ref in response.audio_references:
                    # Provide default book_id if generic (though usually it matches request)
                    book_id = request.book_id if request.book_id else "default"
                    chapter_id = ref.get("chapter_id")

                    # Try mp3 first, then wav (Standardized naming)
                    # {book_id}_chapter{chapter_id:02d}.{ext}
                    blob_name_mp3 = f"uploads/{book_id}/{book_id}_chapter{chapter_id:02d}.mp3"
                    blob_name_wav = f"uploads/{book_id}/{book_id}_chapter{chapter_id:02d}.wav"

                    # We check existence to ensure we don't return broken links
                    blob = bucket.blob(blob_name_mp3)
                    if not blob.exists():
                         blob = bucket.blob(blob_name_wav)

                    # Generate URL if blob exists
                    if blob.exists():
                        try:
                            # Attempt to sign
                            url = blob.generate_signed_url(
                                version="v4",
                                expiration=timedelta(minutes=60),
                                method="GET"
                            )
                            ref["url"] = url
                        except Exception as sign_err:
                            logger.error(f"Failed to sign URL for {blob.name}: {sign_err}") 
                            # Fallback: if we can't sign, maybe the bucket is public? 
                            # ref["url"] = blob.public_url 

            except Exception as e:
                # Log error but return text answer
                logger.error(f"Error connecting to GCS for audio refs: {e}", exc_info=True)
        
        return response
    except Exception as e:
        # Log the full error for admins/developers
        # logger.error(f"QA Error: {e}", exc_info=True)
        # Return a graceful error to the user
        raise HTTPException(status_code=500, detail=f"I encountered an issue answering that. Please try again. Error: {str(e)}")


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
    # GCS UPLOAD HELPER
    # ------------------------------------------------------------
    from google.cloud import storage
    import os

    def upload_to_gcs(local_path: Path, destination_blob_name: str):
        storage_client = storage.Client()
        bucket_name = os.getenv("GCP_BUCKET_NAME", "audioseek-bucket")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(str(local_path))
        return f"gs://{bucket_name}/{destination_blob_name}"

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
            # Skip Mac metadata/resource fork files and __MACOSX directories
            if audio_file.name.startswith("._") or "__MACOSX" in audio_file.parts:
                continue
            
            # Move only files, avoiding self-move if already in root
            if audio_file.is_file() and audio_file.suffix.lower() in [".mp3", ".wav"]:
                if audio_file.parent != extract_dir:
                    shutil.move(str(audio_file), extract_dir)

        # Clean up leftover directories
        for sub in extract_dir.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)
        
        # ---- UPLOAD TO GCS ----
        # Standardized path: uploads/{book_id}/
        # Files renamed to: {book_id}_chapter{i}.{ext}
        book_id = book_name  # book_name is already normalized above
        gcs_prefix = f"uploads/{book_id}"
        
        # Sort files to ensure deterministic order processing
        # Also ensure we don't pick up any stray hidden files
        audio_files = sorted([
            f for f in extract_dir.glob("*") 
            if f.suffix.lower() in [".mp3", ".wav"] and not f.name.startswith("._")
        ])
        
        for i, audio_file in enumerate(audio_files, start=1):
             new_filename = f"{book_id}_chapter{i:02d}{audio_file.suffix}"
             upload_to_gcs(audio_file, f"{gcs_prefix}/{new_filename}")

        # Cleanup local
        shutil.rmtree(upload_dir)

        return {
            "status": "uploaded_zip",
            "book_name": book_name,
            "folder_path": f"gs://{os.getenv('GCP_BUCKET_NAME', 'audioseek-bucket')}/{gcs_prefix}"
        }

    # ------------------------------------------------------------
    # SINGLE AUDIO FILE CASE
    # ------------------------------------------------------------
    elif file.filename.endswith((".mp3", ".wav")):
        audio_dir = upload_dir / book_name
        audio_dir.mkdir(exist_ok=True)
        shutil.move(str(file_path), audio_dir)
        
        # ---- UPLOAD TO GCS ----
        # Standardized path: uploads/{book_id}/
        # File renamed to: {book_id}_chapter1.{ext}
        book_id = book_name
        file_ext = Path(file.filename).suffix
        new_filename = f"{book_id}_chapter01{file_ext}"
        
        gcs_path = f"uploads/{book_id}/{new_filename}"
        
        final_local_path = audio_dir / file.filename
        upload_to_gcs(final_local_path, gcs_path)
        
        # Cleanup local
        shutil.rmtree(upload_dir)

        return {
            "status": "uploaded_audio",
            "book_name": book_name,
            "folder_path": f"gs://{os.getenv('GCP_BUCKET_NAME', 'audioseek-bucket')}/uploads/{book_id}"
        }

    else:
        raise HTTPException(400, "Unsupported file format. Upload .mp3, .wav, or .zip")

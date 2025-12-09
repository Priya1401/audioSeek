import logging
import time
import traceback
from typing import Dict, Any

from domain.models import AudioProcessRequest, ChunkingRequest, EmbeddingRequest, AddFromFilesRequest
from domain.models_firestore import Job, JobStatus
from services.storage.firestore_service import firestore_db
from services.audio.transcription_service import TranscriptionService
from services.nlp.chunking_service import ChunkingService
from services.nlp.embedding_service import EmbeddingService
from services.storage.vector_db_service import VectorDBService
from services.storage.vector_db_service import VectorDBService
from services.storage.metadata_db_service import MetadataDBService
from services.notifications.email_service import email_service
import mlflow
from core.config_mlflow import MLFLOW_EXPERIMENT_NAME

logger = logging.getLogger(__name__)

class JobService:
    def __init__(self):
        self.db = firestore_db
        self.transcription_service = TranscriptionService()
        self.metadata_db = MetadataDBService()

    def create_job(self, request: AudioProcessRequest, user_email: str) -> Job:
        import uuid
        job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            user_email=user_email,
            book_name=request.book_name,
            status=JobStatus.PENDING,
            message="Job created, waiting for processor..."
        )
        
        self.db.create_job(job)
        return job

    def process_audio_background(self, job_id: str, request: AudioProcessRequest):
        """
        Background task to run the full audio processing pipeline.
        """
        logger.info(f"Starting background job {job_id} for book {request.book_name}")
        
        # Start MLflow run
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"process_audio_{request.book_name}"):
            mlflow.log_param("job_id", job_id)
            mlflow.log_param("book_name", request.book_name)
            mlflow.log_param("user_email", request.user_email)
            mlflow.log_param("model_size", request.model_size)
            
            start_total = time.time()
            
            try:
                # Update status to PROCESSING
                self.db.update_job_status(job_id, JobStatus.PROCESSING, message="Starting transcription...", progress=0.1)
                
                # ------------------ STEP 1: TRANSCRIBE ------------------
                start_trans = time.time()
                transcripts = self.transcription_service.transcribe_directory(
                    folder_path=request.folder_path,
                    book_name=request.book_name,
                    model_size=request.model_size,
                    beam_size=request.beam_size,
                    compute_type=request.compute_type
                )
                transcription_time = time.time() - start_trans
                logger.info(f"Transcription complete for job {job_id} in {transcription_time:.2f}s")
                mlflow.log_metric("transcription_time_sec", transcription_time)
                mlflow.log_metric("transcribed_files_count", len(transcripts["files"]))
                
                self.db.update_job_status(job_id, JobStatus.PROCESSING, message="Transcription complete. Chunking...", progress=0.4)

                # ------------------ STEP 2: CHUNK -----------------------
                start_chunk = time.time()
                chunk_output_file = f"/app/raw_data/transcription_results/{request.book_name}/chunks_output.json"

                chunk_request = ChunkingRequest(
                    folder_path=f"/app/raw_data/transcription_results/{request.book_name}",
                    book_id=request.book_name,
                    target_tokens=request.target_tokens,
                    overlap_tokens=request.overlap_tokens,
                    output_file=chunk_output_file
                )

                chunk_resp = ChunkingService.chunk_transcript(chunk_request)
                logger.info(f"Chunking complete for job {job_id}. {len(chunk_resp.chunks)} chunks.")
                mlflow.log_metric("chunking_time_sec", time.time() - start_chunk)
                mlflow.log_metric("chunks_count", len(chunk_resp.chunks))

                # --------------------------------------------------------
                # NEW: Save to Metadata DB (SQLite) for Admin Dashboard
                # --------------------------------------------------------
                try:
                    # 1. Ensure Book Exists
                    self.metadata_db.create_audiobook(book_id=request.book_name, title=request.book_name.replace("_", " ").title())
                    
                    # 2. Save Chapters
                    chapter_ids_map = {}
                    for ch in chunk_resp.chapters:
                        c_num = ch.get("id", 0)
                        cid = self.metadata_db.create_chapter(
                            book_id=request.book_name,
                            chapter_number=c_num,
                            title=ch.get("title", f"Chapter {c_num}"),
                            start_time=ch.get("start_time", 0.0),
                            end_time=ch.get("end_time", 0.0),
                            summary=ch.get("summary")
                        )
                        chapter_ids_map[c_num] = cid
                        
                    # 3. Save Chunks
                    for chunk in chunk_resp.chunks:
                        # map external chapter_id (int from chunking) to internal DB id
                         original_chap_id = chunk.get("chapter_id")
                         internal_chap_id = chapter_ids_map.get(original_chap_id)
                         
                         self.metadata_db.create_chunk(
                            book_id=request.book_name,
                            chapter_id=internal_chap_id,
                            text=chunk["text"],
                            start_time=chunk["start_time"],
                            end_time=chunk["end_time"],
                            token_count=chunk["token_count"],
                            source_file=chunk.get("source_file")
                        )
                    logger.info(f"Metadata saved to SQLite for job {job_id}")
                    
                except Exception as db_err:
                    logger.error(f"Failed to save metadata to SQLite for job {job_id}: {db_err}")
                    # Continue pipeline even if stats fail
                
                self.db.update_job_status(job_id, JobStatus.PROCESSING, message="Chunking complete. Generating embeddings...", progress=0.6)

                # ------------------ STEP 3: EMBED -----------------------
                start_embed = time.time()
                embed_output_file = f"/app/raw_data/transcription_results/{request.book_name}/embeddings_output.json"

                embed_request = EmbeddingRequest(
                    chunks_file=chunk_resp.output_file,
                    book_id=request.book_name,
                    output_file=embed_output_file
                )

                embed_resp = EmbeddingService.generate_embeddings(embed_request)
                logger.info(f"Embedding complete for job {job_id}. {embed_resp.count} embeddings.")
                mlflow.log_metric("embedding_time_sec", time.time() - start_embed)
                mlflow.log_metric("embeddings_count", embed_resp.count)
                
                self.db.update_job_status(job_id, JobStatus.PROCESSING, message="Embeddings generated. Indexing...", progress=0.8)

                # ------------------ STEP 4: VECTOR DB -------------------
                start_vector = time.time()
                add_request = AddFromFilesRequest(
                    chunks_file=chunk_resp.output_file,
                    embeddings_file=embed_resp.output_file,
                    book_id=request.book_name
                )

                add_resp = VectorDBService.add_from_files(add_request)
                logger.info(f"Vector DB indexing complete for job {job_id}.")
                mlflow.log_metric("vector_indexing_time_sec", time.time() - start_vector)
                mlflow.log_metric("total_duration_sec", time.time() - start_total)

                # ------------------ COMPLETE -----------------------
                result = {
                    "book_name": request.book_name,
                    "transcription_count": len(transcripts["files"]),
                    "chunks_count": len(chunk_resp.chunks),
                    "embeddings_count": embed_resp.count,
                    "vector_db_count": add_resp.embeddings_count
                }
                
                self.db.update_job_status(
                    job_id, 
                    JobStatus.COMPLETED, 
                    message="Processing successfully completed.", 
                    progress=1.0, 
                    result=result
                )
                
                # Send email notification
                if request.user_email and "@" in request.user_email:
                    subject = f"AudioSeek Processing Complete: {request.book_name}"
                    body = (
                        f"Hello,\n\n"
                        f"Your audiobook '{request.book_name}' has been successfully processed.\n"
                        f"Stats:\n"
                        f"- Transcribed Files: {len(transcripts['files'])}\n"
                        f"- Chunks: {len(chunk_resp.chunks)}\n"
                        f"- Embeddings: {embed_resp.count}\n\n"
                        f"You can now chat with your book in the AudioSeek interface.\n\n"
                        f"Job ID: {job_id}"
                    )
                    
                    # Prepare BCC admins
                    admin_emails = self._get_admin_emails()
                    email_service.send_notification(request.user_email, subject, body, bcc=admin_emails)

            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                traceback.print_exc()
                mlflow.log_param("error", str(e))
                self.db.update_job_status(
                    job_id, 
                    JobStatus.FAILED, 
                    message=f"Processing failed: {str(e)}", 
                    error=str(e)
                )
                
                # Send failure email
                if request.user_email and "@" in request.user_email:
                    subject = f"AudioSeek Processing Failed: {request.book_name}"
                    body = (
                        f"Hello,\n\n"
                        f"Unfortunately, processing for your audiobook '{request.book_name}' failed.\n"
                        f"Error: {str(e)}\n\n"
                        f"Job ID: {job_id}"
                    )
                    # Prepare BCC admins
                    admin_emails = self._get_admin_emails()
                    email_service.send_notification(request.user_email, subject, body, bcc=admin_emails)

    def _get_admin_emails(self):
        import os
        admin_str = os.getenv("ADMIN_EMAILS", "")
        return [email.strip() for email in admin_str.split(",") if email.strip()]

    def check_stale_jobs(self):
        """
        Check for jobs that have been stuck in PROCESSING for too long and mark them as FAILED.
        This is typically called on application startup.
        """
        logger.info("Checking for stale jobs...")
        try:
            stale_jobs = self.db.get_stale_jobs(threshold_minutes=60)
            
            if not stale_jobs:
                logger.info("No stale jobs found.")
                return
                
            logger.warning(f"Found {len(stale_jobs)} stale jobs. Marking as FAILED.")
            
            for job in stale_jobs:
                error_msg = "Job interrupted by server restart or timeout. Please retry."
                
                self.db.update_job_status(
                    job.job_id,
                    JobStatus.FAILED,
                    message=error_msg,
                    error="StaleJobError"
                )
                logger.info(f"Marked stale job {job.job_id} as FAILED.")
                
        except Exception as e:
            logger.error(f"Failed to check stale jobs: {e}")

# Global instance
job_service = JobService()

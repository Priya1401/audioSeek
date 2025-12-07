import glob
import json
import logging
import os
import sqlite3
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from typing import Dict, Any, List

from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from core.config import settings
# MLflow imports
import mlflow
from core.config_mlflow import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)

from domain.models import (
    AddFromFilesResponse,
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
    QueryResponse
)
from core.utils import (
    parse_transcript,
    detect_chapters,
    chunk_text,
    collect_unique_entities,
    extract_chapter_from_filename,
    extract_book_id_from_path
)

from core.config import settings
from services.storage.vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

from services.nlp.chunking_service import ChunkingService
from services.storage.metadata_db_service import MetadataDBService
from services.storage.vector_db_service import get_vector_db

# Load embedding model
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

from core.model_loader import get_embedding_model

embedding_model = get_embedding_model()

class PipelineService:
    """Full ingestion pipeline"""

    @staticmethod
    def process_combined_pipeline(request: CombinedRequest) -> CombinedResponse:
        book_id = extract_book_id_from_path(
            book_id=request.book_id,
            folder_path=request.folder_path,
            file_path=request.file_path,
            file_paths=request.file_paths
        )
        logger.info(f"Starting combined pipeline for book_id: {book_id}")

        chunk_request = ChunkingRequest(
            book_id=book_id,
            file_path=request.file_path,
            file_paths=request.file_paths,
            folder_path=request.folder_path,
            target_tokens=request.target_tokens,
            overlap_tokens=request.overlap_tokens,
            output_file=request.chunks_output_file
        )
        chunk_response = ChunkingService.chunk_transcript(chunk_request)

        texts = [chunk['text'] for chunk in chunk_response.chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = embedding_model.encode(texts).tolist()
        logger.info("Combined pipeline completed successfully")

        response_data = {
            'book_id': book_id,
            'chunks': chunk_response.chunks,
            'chapters': chunk_response.chapters,
            'entities': chunk_response.entities,
            'embeddings': embeddings,
            'processed_files': chunk_response.processed_files,
            'chunks_output_file': chunk_response.output_file
        }

        if request.embeddings_output_file:
            try:
                embedding_data = {
                    'embeddings': embeddings,
                    'count': len(embeddings)
                }
                with open(request.embeddings_output_file, 'w',
                          encoding='utf-8') as f:
                    json.dump(embedding_data, f, indent=2)
                response_data[
                    'embeddings_output_file'] = request.embeddings_output_file
                logger.info(
                    f"Embeddings saved to {request.embeddings_output_file}")
            except Exception as e:
                raise HTTPException(500,
                                    f"Error saving embeddings file: {str(e)}")

        return CombinedResponse(**response_data)

    @staticmethod
    def process_full_pipeline(
        request: FullPipelineRequest) -> FullPipelineResponse:
        book_id = extract_book_id_from_path(
            book_id=request.book_id,
            folder_path=request.folder_path,
            file_path=request.file_path,
            file_paths=request.file_paths
        )
        logger.info(f"Running full pipeline for book_id={book_id}")


        # -----------------------------
        # 1. Start MLflow run
        # -----------------------------
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # Safety: End any dangling run on this thread
        if mlflow.active_run():
            logger.warning(f"Found active run {mlflow.active_run().info.run_id}, ending it.")
            mlflow.end_run()

        mlflow.start_run(run_name=f"process_full_{book_id}")

        mlflow.log_param("book_id", book_id)
        mlflow.log_param("target_tokens", request.target_tokens)
        mlflow.log_param("overlap_tokens", request.overlap_tokens)
        mlflow.log_param("add_to_vector_db", request.add_to_vector_db)

        start_time_total = time.time()

        # -----------------------------
        # 2. Log raw file list (not content)
        # -----------------------------
        if request.folder_path and os.path.exists(request.folder_path):
            file_list = sorted([
                f for f in os.listdir(request.folder_path)
                if f.endswith(".txt")
            ])
            mlflow.log_dict(
                {"folder": request.folder_path, "files": file_list},
                artifact_file="raw_input_file_list.json"
            )

        metadata_db = MetadataDBService()
        metadata_db.create_audiobook(book_id=book_id, title=book_id)
        t0 = time.time()

        chunk_request = ChunkingRequest(
            book_id=book_id,
            file_path=request.file_path,
            file_paths=request.file_paths,
            folder_path=request.folder_path,
            target_tokens=request.target_tokens,
            overlap_tokens=request.overlap_tokens
        )
        chunk_response = ChunkingService.chunk_transcript(chunk_request)
        t1 = time.time()
        mlflow.log_metric("num_chunks", len(chunk_response.chunks))
        mlflow.log_metric("num_chapters", len(chunk_response.chapters))
        mlflow.log_metric("num_entities", len(chunk_response.entities))
        mlflow.log_metric("time_chunking_sec", t1 - t0)

        # Artifact: chunks
        chunk_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(chunk_artifact.name, "w") as f:
            json.dump(chunk_response.chunks, f, indent=2)
        mlflow.log_artifact(chunk_artifact.name, artifact_path="chunks")

        # Artifact: chapters
        chapter_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(chapter_artifact.name, "w") as f:
            json.dump(chunk_response.chapters, f, indent=2)
        mlflow.log_artifact(chapter_artifact.name, artifact_path="chapters")

        # Artifact: entities
        entity_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(entity_artifact.name, "w") as f:
            json.dump(chunk_response.entities, f, indent=2)
        mlflow.log_artifact(entity_artifact.name, artifact_path="entities")

        # *** CHECK 1: Verify chunks in response ***
        logger.info("=" * 70)
        logger.info("CHECK 1: Chunks in ChunkResponse")
        for i in range(min(3, len(chunk_response.chunks))):
            chunk = chunk_response.chunks[i]
            logger.info(
                f"  Chunk {i}: chapter_id={chunk.get('chapter_id')}, type={type(chunk)}")

        null_count = sum(
            1 for c in chunk_response.chunks if c.get('chapter_id') is None)
        logger.info(
            f"  Total chunks: {len(chunk_response.chunks)}, Null chapter_ids: {null_count}")
        logger.info("=" * 70)

        # Save chapters
        chapter_ids = {}
        for chapter in chunk_response.chapters:
            chapter_number = chapter.get("id", 0)
            cid = metadata_db.create_chapter(
                book_id=book_id,
                chapter_number=chapter_number,
                title=chapter.get("title", f"Chapter {chapter_number}"),
                start_time=chapter.get("start_time", 0.0),
                end_time=chapter.get("end_time", 0.0),
                summary=chapter.get("summary")
            )
            chapter_ids[chapter.get("id")] = cid

        # Save chunks
        for chunk in chunk_response.chunks:
            metadata_db.create_chunk(
                book_id=book_id,
                chapter_id=chapter_ids.get(chunk.get("chapter_id")),
                text=chunk["text"],
                start_time=chunk["start_time"],
                end_time=chunk["end_time"],
                token_count=chunk["token_count"],
                source_file=chunk.get("source_file")
            )

        t2 = time.time()
        texts = [c["text"] for c in chunk_response.chunks]
        embeddings = embedding_model.encode(texts).tolist()
        t3 = time.time()

        mlflow.log_metric("embedding_count", len(embeddings))
        mlflow.log_metric("time_embedding_sec", t3 - t2)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

        # embeddings.npy artifact
        embed_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(embed_file.name, np.array(embeddings))
        mlflow.log_artifact(embed_file.name, artifact_path="embeddings")

        vector_db_added = False
        if request.add_to_vector_db:
            vector_db = get_vector_db(book_id=book_id)

            metadatas = [
                {
                    "text": c["text"],
                    "start_time": c["start_time"],
                    "end_time": c["end_time"],
                    "chapter_id": c.get("chapter_id"),
                    "token_count": c["token_count"],
                    "source_file": c.get("source_file"),
                }
                for c in chunk_response.chunks
            ]
            t4 = time.time()
            # *** CHECK 2: Verify metadatas before sending ***
            logger.info("=" * 70)
            logger.info("CHECK 2: Metadatas before Vector DB")
            for i in range(min(3, len(metadatas))):
                meta = metadatas[i]
                logger.info(
                    f"  Metadata {i}: chapter_id={meta.get('chapter_id')}, source={meta.get('source_file')}")

            null_meta_count = sum(
                1 for m in metadatas if m.get('chapter_id') is None)
            logger.info(
                f"  Total metadatas: {len(metadatas)}, Null chapter_ids: {null_meta_count}")
            logger.info("=" * 70)

            vector_db.add_documents(embeddings, metadatas)
            t5 = time.time()

            vector_db_added = True
            mlflow.log_metric("time_vector_db_write_sec", t5 - t4)

            # Log FAISS artifacts
            mlflow.log_artifact(vector_db.index_file, artifact_path="faiss")
            mlflow.log_artifact(vector_db.metadata_file, artifact_path="faiss")

            mlflow.log_metric("faiss_index_size_mb", os.path.getsize(vector_db.index_file) / 1e6)
            mlflow.log_metric("faiss_metadata_size_mb", os.path.getsize(vector_db.metadata_file) / 1e6)

        # -----------------------------
        # 6. Plot: Chunk Token Distribution
        # -----------------------------
        token_counts = [c["token_count"] for c in chunk_response.chunks]
        plt.figure(figsize=(8, 5))
        plt.hist(token_counts, bins=30, color="skyblue", edgecolor="black")
        plt.title("Chunk Token Count Distribution")
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")

        token_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(token_plot.name)
        mlflow.log_artifact(token_plot.name, artifact_path="plots")
        plt.close()

        # -----------------------------
        # Plot: Chapter Duration Distribution
        # -----------------------------
        chapter_durations = [
            c["end_time"] - c["start_time"]
            for c in chunk_response.chapters if c["end_time"] and c["start_time"]
        ]

        if chapter_durations:
            plt.figure(figsize=(8, 5))
            plt.hist(chapter_durations, bins=20, color="orange", edgecolor="black")
            plt.title("Chapter Duration Distribution")
            plt.xlabel("Seconds")
            plt.ylabel("Frequency")

            chapter_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(chapter_plot.name)
            mlflow.log_artifact(chapter_plot.name, artifact_path="plots")
            plt.close()

        # -----------------------------
        # Plot: Entity Frequency
        # -----------------------------
        if chunk_response.entities:
            entity_names = [e["name"] for e in chunk_response.entities]
            plt.figure(figsize=(10, 5))
            sns.countplot(x=entity_names)
            plt.xticks(rotation=45, ha="right")
            plt.title("Entity Count Distribution")

            ent_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(ent_plot.name)
            mlflow.log_artifact(ent_plot.name, artifact_path="plots")
            plt.close()

        # -----------------------------
        # Plot: Embedding Similarity Heatmap
        # -----------------------------
        try:
            emb_matrix = np.array(embeddings[:60])  # limit to avoid massive matrices
            if emb_matrix.shape[0] > 2:
                sim_matrix = np.inner(emb_matrix, emb_matrix)

                plt.figure(figsize=(10, 8))
                sns.heatmap(sim_matrix, cmap="viridis")
                plt.title("Embedding Similarity Heatmap")

                sim_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                plt.savefig(sim_plot.name)
                mlflow.log_artifact(sim_plot.name, artifact_path="plots")
                plt.close()
        except Exception as e:
            logger.error(f"Error creating similarity heatmap: {e}")

        # -----------------------------
        # 7. Log DB snapshot
        # -----------------------------
        metadata_db_path = "audiobook_metadata.db"
        if os.path.exists(metadata_db_path):
            pass
            mlflow.log_artifact(metadata_db_path, artifact_path="db_snapshot")

        # -----------------------------
        # 8. Finish MLflow Run
        # -----------------------------
        end_time_total = time.time()
        mlflow.log_metric("total_pipeline_time_sec", end_time_total - start_time_total)
        mlflow.end_run()

        # -----------------------------
        # 9. Return Response
        # -----------------------------
        return FullPipelineResponse(
            book_id=book_id,
            chunks_count=len(chunk_response.chunks),
            chapters_count=len(chunk_response.chapters),
            entities_count=len(chunk_response.entities),
            embeddings_count=len(embeddings),
            vector_db_added=vector_db_added,
            files_processed=len(chunk_response.processed_files),
            message=f"Full pipeline completed for book_id={book_id}"
        )

        mlflow.end_run()
        return response
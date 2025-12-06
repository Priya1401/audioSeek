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
from services.storage.faiss_vector_db import FAISSVectorDB
from services.storage.gcp_vector_db import GCPVectorDB

logger = logging.getLogger(__name__)

# Load embedding model
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

from core.model_loader import get_embedding_model

embedding_model = get_embedding_model()

def get_vector_db(book_id: str = "default") -> VectorDBInterface:
    if settings.vector_db_type == "gcp":
        return GCPVectorDB(
            project_id=settings.gcp_project_id,
            bucket_name=settings.gcp_bucket_name,
            book_id=book_id
        )
    else:
        return FAISSVectorDB(
            book_id=book_id,
            bucket_name=settings.gcp_bucket_name,
            project_id=settings.gcp_project_id
        )

class VectorDBService:

    @staticmethod
    def add_from_files(request) -> AddFromFilesResponse:
        book_id = extract_book_id_from_path(
            book_id=request.book_id,
            chunks_file=request.chunks_file
        )
        logger.info(f"Adding documents for book_id: {book_id}")

        with open(request.chunks_file, 'r') as f:
            chunks_data = json.load(f)
        chunks = chunks_data["chunks"]

        with open(request.embeddings_file, 'r') as f:
            embed_data = json.load(f)
        embeddings = embed_data["embeddings"]

        if len(chunks) != len(embeddings):
            raise HTTPException(400, "Mismatch in chunks/embeddings length")

        metadatas = [
            {
                "text": c["text"],
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "chapter_id": c.get("chapter_id"),
                "token_count": c["token_count"],
                "source_file": c.get("source_file")
            }
            for c in chunks
        ]

        vector_db = get_vector_db(book_id=book_id)
        vector_db.add_documents(embeddings, metadatas)

        return AddFromFilesResponse(
            message=f"Added {len(chunks)} documents for {book_id}",
            chunks_count=len(chunks),
            embeddings_count=len(embeddings)
        )

    @staticmethod
    def add_documents(request: AddDocumentsRequest) -> AddDocumentsResponse:
        book_id = request.book_id if request.book_id else "default"
        logger.info(
            f"Adding {len(request.embeddings)} documents for book_id: {book_id}")

        if len(request.embeddings) != len(request.metadatas):
            raise HTTPException(400, "Embeddings/metadatas mismatch")

        vector_db = get_vector_db(book_id=book_id)
        result = vector_db.add_documents(request.embeddings, request.metadatas)

        return AddDocumentsResponse(
            message=result.get("message", "Added documents"),
            count=result.get("count", len(request.embeddings))
        )

    @staticmethod
    def search(request: SearchRequest) -> SearchResponse:
        book_id = request.book_id if request.book_id else "default"
        logger.info(f"Searching in book_id: {book_id}")

        vector_db = get_vector_db(book_id=book_id)
        results = vector_db.search(request.query_embedding, request.top_k)
        return SearchResponse(results=results, count=len(results))

    @staticmethod
    def query_text(request: QueryRequest) -> SearchResponse:
        book_id = request.book_id if request.book_id else "default"
        logger.info(f"Querying text in book_id: {book_id}")

        query_embedding = embedding_model.encode([request.query])[0].tolist()
        vector_db = get_vector_db(book_id=book_id)
        results = vector_db.search(query_embedding, request.top_k)
        return SearchResponse(results=results, count=len(results))

    @staticmethod
    def get_stats(book_id: str = "default"):
        vector_db = get_vector_db(book_id=book_id)
        return vector_db.get_stats()

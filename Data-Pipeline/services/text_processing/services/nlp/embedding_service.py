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

# Load embedding model
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

from core.model_loader import get_embedding_model

embedding_model = get_embedding_model()

class EmbeddingService:
    """Service for generating embeddings"""

    @staticmethod
    def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            book_id = extract_book_id_from_path(
                book_id=request.book_id,
                chunks_file=request.chunks_file
            )
            logger.info(f"Generating embeddings for book_id: {book_id}")

            if request.chunks_file:
                if not os.path.exists(request.chunks_file):
                    raise HTTPException(404,
                                        f"File not found: {request.chunks_file}")

                with open(request.chunks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                texts = [c["text"] for c in data.get("chunks", [])]

            elif request.texts:
                texts = request.texts
            else:
                raise HTTPException(400, "Provide 'texts' or 'chunks_file'")

            if not texts:
                raise HTTPException(400, "No texts found")

            logger.info(f"Generating Embedding for Chunks using {embedding_model} ")

            embeddings_np = embedding_model.encode(
                        texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # optional but recommended
            )


            logger.info(f"Shape generated: {embeddings_np.shape}")

            embeddings = embeddings_np.tolist()

            response = {
                "embeddings": embeddings,
                "count": len(embeddings),
                "book_id": book_id
            }

            if request.output_file:
                with open(request.output_file, 'w', encoding='utf-8') as f:
                    json.dump(response, f, indent=2)
                response["output_file"] = request.output_file

            return EmbeddingResponse(**response)

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(500, str(e))

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

class ChunkingService:
    """Service for chunking transcripts"""

    @staticmethod
    def _get_file_list(request: ChunkingRequest) -> List[str]:
        files = []

        if request.file_path:
            if not os.path.exists(request.file_path):
                raise HTTPException(404, f"File not found: {request.file_path}")
            files = [request.file_path]

        # Multiple files
        elif request.file_paths:
            for fp in request.file_paths:
                if not os.path.exists(fp):
                    raise HTTPException(404, f"File not found: {fp}")
            files = request.file_paths

        # Folder
        elif request.folder_path:
            if not os.path.exists(request.folder_path):
                raise HTTPException(404,
                                    f"Folder not found: {request.folder_path}")

            pattern = os.path.join(request.folder_path, "*.txt")
            files = glob.glob(pattern)

            if not files:
                raise HTTPException(404,
                                    f"No .txt files found in folder: {request.folder_path}")

        for f in files:
            if not f.endswith(".txt"):
                raise HTTPException(400, f"Only .txt files allowed: {f}")

        return files

    @staticmethod
    def _process_single_file(file_path: str, target_tokens: int,
        overlap_tokens: int) -> Dict[str, Any]:
        logger.info(f"Processing file: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

        segments = parse_transcript(transcript)

        if not segments:
            logger.warning(f"No valid segments found in {file_path}")
            return {
                'chunks': [],
                'chapters': [],
                'entities': [],
                'file': file_path
            }

        chapters = detect_chapters(segments)

        filename = os.path.basename(file_path)
        fallback_chapter_id = extract_chapter_from_filename(filename)

        if fallback_chapter_id:
            logger.info(
                f"Extracted chapter {fallback_chapter_id} from filename: {filename}")
        else:
            logger.warning(
                f"Could not extract chapter from filename: {filename}")

        if not chapters and fallback_chapter_id:
            chapters = [{
                'id': fallback_chapter_id,
                'title': f'Chapter {fallback_chapter_id}',
                'start_time': segments[0]['start'],
                'end_time': segments[-1]['end']
            }]
            logger.info(
                f"Created chapter entry from filename: Chapter {fallback_chapter_id}")

        logger.info(
            f"About to chunk with fallback_chapter_id={fallback_chapter_id}")
        chunks = chunk_text(segments, target_tokens, overlap_tokens, chapters,
                            fallback_chapter_id)
        logger.info(f"Generated {len(chunks)} chunks from chunk_text()")

        # *** FORCE chapter_id for all chunks from this file ***
        chunks_before_fix = sum(
            1 for c in chunks if c.get('chapter_id') is None)
        logger.info(
            f"Chunks with null chapter_id BEFORE fix: {chunks_before_fix}/{len(chunks)}")

        for i, chunk in enumerate(chunks):
            chunk['source_file'] = os.path.basename(file_path)

            # Override chapter_id with fallback if we have one
            if fallback_chapter_id is not None:
                old_chapter_id = chunk.get('chapter_id')
                chunk['chapter_id'] = fallback_chapter_id
                logger.info(
                    f"Chunk {i}: Set chapter_id from {old_chapter_id} to {fallback_chapter_id} at {chunk['start_time']:.1f}s")

        chunks_after_fix = sum(1 for c in chunks if c.get('chapter_id') is None)
        logger.info(
            f"Chunks with null chapter_id AFTER fix: {chunks_after_fix}/{len(chunks)}")

        entities = collect_unique_entities(chunks)

        logger.info(
            f"Returning {len(chunks)} chunks, all should have chapter_id={fallback_chapter_id}")

        return {
            'chunks': chunks,
            'chapters': chapters,
            'entities': entities,
            'file': file_path
        }

    @staticmethod
    def chunk_transcript(request: ChunkingRequest) -> ChunkResponse:
        # Extract book_id
        book_id = extract_book_id_from_path(
            book_id=request.book_id,
            folder_path=request.folder_path,
            file_path=request.file_path,
            file_paths=request.file_paths
        )
        logger.info(f"Processing with book_id: {book_id}")

        files = ChunkingService._get_file_list(request)
        logger.info(f"Processing {len(files)} file(s)")

        all_chunks, all_chapters = [], []
        all_entities = {}
        processed_files = []

        for fp in files:
            try:
                result = ChunkingService._process_single_file(
                    fp,
                    request.target_tokens,
                    request.overlap_tokens
                )

                all_chunks.extend(result["chunks"])
                all_chapters.extend(result["chapters"])

                for entity in result["entities"]:
                    key = (entity["name"], entity["type"])
                    if key not in all_entities:
                        all_entities[key] = entity

                processed_files.append(fp)

            except Exception as e:
                logger.error(f"Error processing {fp}: {e}")
                continue

        if not all_chunks:
            raise HTTPException(400, "No valid chunks generated.")

        entities_list = list(all_entities.values())

        response = {
            "book_id": book_id,
            "chunks": all_chunks,
            "chapters": all_chapters,
            "entities": entities_list,
            "processed_files": processed_files
        }

        # Optional: save output file
        if request.output_file:
            with open(request.output_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2)
            response["output_file"] = request.output_file

        return ChunkResponse(**response)

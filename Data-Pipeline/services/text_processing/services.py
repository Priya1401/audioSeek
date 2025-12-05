import glob
import json
import logging
import os
import sqlite3
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import time
from abc import ABC, abstractmethod
from groq import Groq

from typing import Dict, Any, List

from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from config import settings
# MLflow imports
import mlflow
from config_mlflow import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME
)

from models import (
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
from utils import (
    parse_transcript,
    detect_chapters,
    chunk_text,
    collect_unique_entities,
    extract_chapter_from_filename,
    extract_book_id_from_path
)

from config import settings
from vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

# Load embedding model
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = SentenceTransformer(
    "BAAI/bge-m3",
    trust_remote_code=True
)

# Multi-book vector DB instances
_vector_db_instances: Dict[str, VectorDBInterface] = {}


def get_vector_db(book_id: str = "default") -> VectorDBInterface:
    """Get or initialize the vector database instance for a specific book_id"""
    global _vector_db_instances

    if book_id in _vector_db_instances:
        return _vector_db_instances[book_id]

    # Create new instance
    if settings.vector_db_type.lower() == 'gcp':
        from gcp_vector_db import GCPVectorDB
        logger.info(f"Initializing GCP Vector DB for book_id={book_id}")

        instance = GCPVectorDB(
            project_id=settings.gcp_project_id,
            location=settings.gcp_region,
            index_id=f"{settings.gcp_index_id}-{book_id}",
            index_endpoint_id=settings.gcp_index_endpoint_id,
            credentials_path=settings.gcp_credentials_path
        )

        if not instance.verify_connection():
            logger.warning(
                f"GCP Vector DB connection failed for book_id={book_id}")

    else:
        # Local FAISS (default)
        from faiss_vector_db import FAISSVectorDB
        logger.info(f"Initializing FAISS Vector DB for book_id={book_id}")

        instance = FAISSVectorDB(
            book_id=book_id,
            bucket_name=settings.gcp_bucket_name,
            project_id=settings.gcp_project_id
        )

    _vector_db_instances[book_id] = instance
    return instance


# ------------------------------------------------------------
# CHUNKING SERVICE (unchanged except safety improvements)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# EMBEDDING SERVICE (unchanged)
# ------------------------------------------------------------
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
                "count": len(embeddings)
            }

            if request.output_file:
                with open(request.output_file, 'w', encoding='utf-8') as f:
                    json.dump(response, f, indent=2)
                response["output_file"] = request.output_file

            return EmbeddingResponse(**response)

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(500, str(e))


# ------------------------------------------------------------
# VECTOR DB SERVICE (Fix 3 applied)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# METADATA DATABASE SERVICE (Fix 5)
# ------------------------------------------------------------
import sqlite3

class MetadataDBService:
    """Book-aware metadata database operations"""

    def __init__(self, db_path="audiobook_metadata.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize DB using schema.sql. If missing, fallback schema is created."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Use absolute path relative to this file
            schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
            with open(schema_path, "r") as f:
                schema = f.read()
            cursor.executescript(schema)
            conn.commit()
            logger.info("Metadata DB initialized from schema.sql")
        except Exception as e:
            logger.error(f"Error loading schema.sql: {e}")
            conn.rollback()
        finally:
            conn.close()

    def create_audiobook(self, book_id: str, title: str, author=None,
        duration=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       INSERT
                       OR IGNORE INTO audiobooks (book_id, title, author, duration)
            VALUES (?, ?, ?, ?)
                       """, (book_id, title, author, duration))

        conn.commit()
        conn.close()

    # ---------------------------
    # CHAPTERS
    # ---------------------------
    def create_chapter(self, book_id: str, chapter_number: int,
        title: str, start_time: float,
        end_time: float, summary=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       INSERT INTO chapters (book_id, chapter_number, title,
                                             start_time, end_time, summary)
                       VALUES (?, ?, ?, ?, ?, ?)
                       """,
                       (book_id, chapter_number, title, start_time, end_time,
                        summary))

        chapter_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chapter_id

    def get_chapters(self, book_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT id,
                              chapter_number,
                              title,
                              start_time,
                              end_time,
                              summary
                       FROM chapters
                       WHERE book_id = ?
                       ORDER BY chapter_number
                       """, (book_id,))

        rows = cursor.fetchall()
        conn.close()

        return {"chapters": rows}

    # ---------------------------
    # CHUNKS
    # ---------------------------
    def create_chunk(self, book_id: str, chapter_id: int,
        text: str, start_time: float, end_time: float,
        token_count: int, source_file: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       INSERT INTO chunks (book_id, chapter_id, start_time,
                                           end_time,
                                           text, token_count, source_file)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       """, (book_id, chapter_id, start_time, end_time,
                             text, token_count, source_file))

        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chunk_id

    def get_chunks(self, book_id: str, chapter_id: int = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if chapter_id:
            cursor.execute("""
                           SELECT id,
                                  chapter_id,
                                  start_time,
                                  end_time,
                                  text,
                                  token_count
                           FROM chunks
                           WHERE book_id = ?
                             AND chapter_id = ?
                           ORDER BY start_time
                           """, (book_id, chapter_id))
        else:
            cursor.execute("""
                           SELECT id,
                                  chapter_id,
                                  start_time,
                                  end_time,
                                  text,
                                  token_count
                           FROM chunks
                           WHERE book_id = ?
                           ORDER BY start_time
                           """, (book_id,))

        rows = cursor.fetchall()
        conn.close()

        return {"chunks": rows}

    # ---------------------------
    # ENTITIES
    # ---------------------------
    def create_entity(self, book_id: str, name: str, type: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       INSERT INTO entities (book_id, name, type)
                       VALUES (?, ?, ?)
                       """, (book_id, name, type))

        entity_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return entity_id

    def get_entities(self, book_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT id, name, type
                       FROM entities
                       WHERE book_id = ?
                       """, (book_id,))

        rows = cursor.fetchall()
        conn.close()

        return {"entities": rows}

    # ---------------------------
    # SESSIONS & CHAT HISTORY
    # ---------------------------
    def create_session(self):
        """Create a new session and return its ID"""
        import uuid
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
        conn.commit()
        conn.close()
        return session_id

    def add_chat_message(self, session_id: str, role: str, content: str):
        """Add a message to the chat history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
        """, (session_id, role, content))
        conn.commit()
        conn.close()

    def get_chat_history(self, session_id: str, limit: int = 10):
        """Get recent chat history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content
            FROM chat_history
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        # Return as list of dicts
        return [{"role": r[0], "content": r[1]} for r in rows]


# ------------------------------------------------------------
# QA SERVICE (Fix 2)
# ------------------------------------------------------------

import google.generativeai as genai
class LLMProvider(ABC):
    @abstractmethod
    def generate_answer(self, prompt, **kwargs):
        pass

class GeminiProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-flash-latest')

    def generate_answer(self, prompt, **kwargs):
        return self.llm.generate_content(prompt, **kwargs).text

class LlamaProvider(LLMProvider):
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = "llama-3.1-8b-instant"
        # Check if setup is successful
        logger.info(f"✓ Groq client initialized with model: {self.model}")

    def generate_answer(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

class QAService:
    """
    FINAL BASELINE QA PIPELINE
    - Uses LLM (S2) for question classification
    - Supports:
        • identity questions
        • event questions
        • first-meet questions (Option A: prev + hit + next)
        • chapter summary
        • timestamp summary
        • general questions
        • session/history questions (NO RETRIEVAL)
    """

    def __init__(self, metadata_db: MetadataDBService, llm_provider: LLMProvider):
        self.metadata_db = metadata_db
        self.provider = llm_provider

    # ======================================================
    # 1) LLM QUESTION CLASSIFIER (S2)
    # ======================================================
    def classify_question_with_llm(self, query: str) -> dict:
        """
        Uses LLM to classify user intent.
        Returns:
        {
          "question_type": "...",
          "entities": [...],
          "chapter": null,
          "start_seconds": null,
          "end_seconds": null,
          "exclude_last_seconds": null
        }
        """

        prompt = f"""
        You classify audiobook questions. Return ONLY valid JSON.

        Allowed question_type:
        - "identity"            (Who is Juliet?)
        - "event"               (How does Paris die?)
        - "first_meet"          (When did Harry first meet Hermione?)
        - "chapter_summary"     (Summarize chapter 3)
        - "timestamp_summary"   (Summarize first 20 minutes of chapter 3)
        - "session"             (What did I ask previously? Repeat your answer.)
        - "general"

        Extract:
        - entities: names/places mentioned
        - chapter: number if relevant
        - start_seconds: integer or null
        - end_seconds: integer or null
        - exclude_last_seconds: integer or null

        QUESTION:
        "{query}"

        JSON ONLY:
        """

        try:
            raw = self.provider.generate_answer(prompt)
            cleaned = raw.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)

            # Ensure fields exist
            for f in ["question_type", "entities", "chapter",
                      "start_seconds", "end_seconds", "exclude_last_seconds"]:
                parsed.setdefault(f, None)
            if parsed["entities"] is None:
                parsed["entities"] = []

            logger.info(f"[CLASSIFIER OUTPUT] {parsed}")
            return parsed

        except Exception as e:
            logger.error(f"Classifier failed: {e}")
            return {
                "question_type": "general",
                "entities": [],
                "chapter": None,
                "start_seconds": None,
                "end_seconds": None,
                "exclude_last_seconds": None
            }

    # ======================================================
    # 2) Timestamp + chapter summary retrieval helpers
    # ======================================================
    def retrieve_by_chapter(self, book_id, chapter):
        chunks = self.metadata_db.get_chunks(book_id)["chunks"]
        return [c for c in chunks if c[1] == chapter]

    def retrieve_by_timestamp(self, book_id, start_s, end_s):
        chunks = self.metadata_db.get_chunks(book_id)["chunks"]
        out = []
        for c in chunks:
            st, en = c[2], c[3]
            if end_s is None:
                if st <= start_s <= en:
                    out.append(c)
            else:
                if en >= start_s and st <= end_s:
                    out.append(c)
        return out

    # ======================================================
    # 3) FIRST-MEET RETRIEVAL (Option A)
    # ======================================================
    def retrieve_first_meet(self, entity, all_results):
        """
        Option A: simplest, cleanest baseline.
        - Find FIRST chunk mentioning entity.
        - Return: previous, the hit, next.
        """

        entity = entity.lower()
        hit_index = None

        for i, r in enumerate(all_results):
            ents = r["metadata"].get("entities", [])
            if entity in [e.lower() for e in ents]:
                hit_index = i
                break

        if hit_index is None:
            logger.info("[FIRST_MEET] No entity match found")
            return []

        selected = []

        # previous
        if hit_index - 1 >= 0:
            selected.append(all_results[hit_index - 1])
        # hit
        selected.append(all_results[hit_index])
        # next
        if hit_index + 1 < len(all_results):
            selected.append(all_results[hit_index + 1])

        logger.info(f"[FIRST_MEET] Returning {len(selected)} chunks")
        return selected

    # ======================================================
    # 4) Vector DB retrieval for general/event/identity
    # ======================================================
    def retrieve_with_vector_db(self, vector_db, expanded_queries, top_k):
        """Handles multi-query vector search and merges results."""

        all_results = []

        for qtext in expanded_queries:
            qvec = embedding_model.encode(qtext)
            res = vector_db.search(qvec, top_k)
            all_results.extend(res)

        if not all_results:
            logger.info("[VDB] No matches")
            return []

        # Sort by FAISS cosine score
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

        logger.info(f"[VDB] Total retrieved raw: {len(all_results)}")
        return all_results

    # ---------------------------
    # 5) LLM PROMPT GENERATORS - Created new function to generate more similar queries to existing query
    # ---------------------------

    def expand_query(self, query: str) -> List[str]:
        """"Generate query variations for better retrieval """

        # Using the same model to generate variations of prompts given by user

        # Skip expansion for very simple queries
        if len(query.split()) < 3:
            logger.info("Simple Query, skipping expansion!")
            return [query]

        expansion_prompt = f"""Generate 4 alternative phrasings of this question that mean the same thing: "{query}"
        Requirements:
        - Use different verbs
        - Use different contexts if applicable
        - Keep the core meaning
        - Make them search-friendly
        Return ONLY 4 variations, one per line, no numbering.
        """

        try:
            #response = self.llm.generate_content(expansion_prompt)
            response = self.provider.generate_answer(expansion_prompt)
            #variations = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            variations = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Add original query
            all_queries = [query] + variations[:4]  # Original + 4 variations = 5 total
            return all_queries

        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [query]  # Fallback to original query


    # ---------------------------
    # 6) LLM ANSWER GENERATION
    # ---------------------------
    def generate_answer(self, query: str, context_texts: List[str], chat_history: List[Dict[str, str]] = None):
        context = "\n".join(
            [f"Passage {i + 1}:\n{text}\n" for i, text in
             enumerate(context_texts)]
        )

        history_context = ""
        if chat_history:
            history_context = "\nPrevious Chat History:\n" + "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history]
            ) + "\n"

        # prompt = f"Answer based on the context only:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        # implementing a less strict prompt allowing the llm to make reasonable inferences
        prompt = f"""You are a helpful assistant answering questions about an audiobook.

        INSTRUCTIONS:
        1. Answer using the information from the provided context passages
        2. You may make reasonable inferences based on character relationships and interactions described
        3. If characters support each other, speak intimately, or consistently help one another, you can infer close friendship
        4. Be helpful and direct - don't be overly cautious
        5. If information truly isn't in the context, say so clearly
        6. Use the previous chat history to understand context if the user refers to previous turns.
        7. IMPORTANT: If you answer the question based SOLELY on the chat history (e.g., "What did I ask before?"), start your response with "[NO_CONTEXT] ".
        8. Write in clear, modern, simple English. Avoid poetic, archaic, dramatic, or novel-like language. Rewrite any stylistic text into plain, natural, everyday English.
        9. If the context contains poetic, archaic, or dramatic wording, DO NOT quote it directly. Instead, express the meaning in plain, simple English.

        {history_context}

        Context Passages:
        {context}

        Question: {query}

        Answer:"""
        try:
            #response = self.llm.generate_content(prompt)
            response = self.provider.generate_answer(prompt)
            #return response.text
            return response
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error generating answer: {e}"
    # ======================================================
    # 7) MAIN QA HANDLER
    # ======================================================
    def ask_question(self, request: QueryRequest):

        book_id = request.book_id or "default"
        logger.info(f"[QA] Query: {request.query}")

        # --- Session handling ---
        session_id = request.session_id or self.metadata_db.create_session()
        chat_history = self.metadata_db.get_chat_history(session_id)

        # --- Classify question ---
        cls = self.classify_question_with_llm(request.query)
        qtype = cls["question_type"]
        entities = [e.lower() for e in cls["entities"] if isinstance(e, str)]

        logger.info(f"[QA] Classified as: {qtype}, Entities={entities}")

        # --- Session/history queries: NO RETRIEVAL ---
        if qtype == "session":
            logger.info("[SESSION MODE] No retrieval, answering from chat history")
            answer = self.generate_answer(
                "[SESSION QUERY] " + request.query,
                context_texts=[],
                chat_history=chat_history
            )
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(session_id, "assistant", answer)
            return QueryResponse(answer=answer, citations=[], session_id=session_id)

        # --- Vector DB ready ---
        vector_db = get_vector_db(book_id=book_id)

        # Expand query
        if qtype in ["first_meet", "session", "chapter_summary", "timestamp_summary"]:
            expanded = [request.query]
        else:
            expanded = self.expand_query(request.query)

        logger.info(f"[EXPANDED] {expanded}")

        # --- Retrieve using vector search ---
        raw_results = self.retrieve_with_vector_db(vector_db, expanded, request.top_k)

        # --- Handle specialized retrieval ---
        # 1) First-meet questions
        if qtype == "first_meet" and entities:
            logger.info("[MODE] FIRST_MEET retrieval")
            final_results = self.retrieve_first_meet(entities[0], raw_results)

        # 2) Chapter summary
        elif qtype == "chapter_summary" and cls["chapter"]:
            logger.info("[MODE] CHAPTER SUMMARY retrieval")
            final_results = self.retrieve_by_chapter(book_id, cls["chapter"])

        # 3) Timestamp summary
        elif qtype == "timestamp_summary":
            start = cls["start_seconds"]
            end = cls["end_seconds"]
            logger.info(f"[MODE] TIMESTAMP SUMMARY retrieval: {start}–{end}")
            final_results = self.retrieve_by_timestamp(book_id, start, end)

        # 4) General / identity / event → fallback to vector DB
        else:
            logger.info("[MODE] GENERAL VECTOR RETRIEVAL")
            final_results = raw_results[:7]

        # Build text passages
        if isinstance(final_results, list) and final_results and isinstance(final_results[0], dict):
            # From vector retrieval
            passages = [r["metadata"]["text"] for r in final_results]
            citations = [
                f"Chapter {r['metadata'].get('chapter_id')} | {r['metadata'].get('start_time')} - {r['metadata'].get('end_time')}"
                for r in final_results
            ]
        else:
            # From metadata retrieval
            passages = [c[4] if len(c) > 4 else str(c) for c in final_results]
            citations = ["(chapter/time-based retrieval)"]

        logger.info(f"[FINAL PASSAGES] {len(passages)}")

        # Produce final answer
        answer = self.generate_answer(request.query, passages, chat_history)

        # Save chat history
        self.metadata_db.add_chat_message(session_id, "user", request.query)
        self.metadata_db.add_chat_message(session_id, "assistant", answer)

        return QueryResponse(answer=answer, citations=citations, session_id=session_id)


# ------------------------------------------------------------
# PIPELINE SERVICE (Fix 4)
# ------------------------------------------------------------
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
        # 2. Log raw file list
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

        # Run chunking
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

        # Save artifacts
        chunk_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(chunk_artifact.name, "w") as f:
            json.dump(chunk_response.chunks, f, indent=2)
        mlflow.log_artifact(chunk_artifact.name, artifact_path="chunks")

        chapter_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(chapter_artifact.name, "w") as f:
            json.dump(chunk_response.chapters, f, indent=2)
        mlflow.log_artifact(chapter_artifact.name, artifact_path="chapters")

        entity_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(entity_artifact.name, "w") as f:
            json.dump(chunk_response.entities, f, indent=2)
        mlflow.log_artifact(entity_artifact.name, artifact_path="entities")

        # -----------------------------
        # Save chapters to DB
        # -----------------------------
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

        # Save chunks to DB
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
        logger.info("[CHUNKING] Complete !! Hurray")
        # -----------------------------
        # Embed chunks
        # -----------------------------
        t2 = time.time()
        texts = [c["text"] for c in chunk_response.chunks]
        embeddings = embedding_model.encode(texts).tolist()
        t3 = time.time()

        mlflow.log_metric("embedding_count", len(embeddings))
        mlflow.log_metric("time_embedding_sec", t3 - t2)
        mlflow.log_param("embedding_model", "BAAI/bge-m3")

        embed_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(embed_file.name, np.array(embeddings))
        mlflow.log_artifact(embed_file.name, artifact_path="embeddings")

        vector_db_added = False
        logger.info("[EMBEDDING] Complete !! Hurray")

        # -----------------------------
        #  ADD TO VECTOR DB
        # -----------------------------
        if request.add_to_vector_db:
            vector_db = get_vector_db(book_id=book_id)

            # --------------------------------------------------------------------
            # CHANGE START: Enhanced Metadata Construction
            # --------------------------------------------------------------------

            metadatas = []
            scene_counter = 0
            last_chapter = None


            for idx, c in enumerate(chunk_response.chunks):

                chapter = c.get("chapter_id")

                # Detect scene changes
                if last_chapter != chapter:
                    scene_counter += 1
                    last_chapter = chapter
                elif idx > 0:
                    prev = chunk_response.chunks[idx - 1]
                    time_gap = c["start_time"] - prev["end_time"]
                    # Check for time gap to isolate scenes helps with precise isolation of
                    if time_gap > 20:
                        scene_counter += 1

                # FIX: Get entities directly from chunk_text output
                chunk_entities = c.get("entities", [])

                # Extract just the names
                entity_names = [
                    e["name"] for e in chunk_entities
                    if isinstance(e, dict) and "name" in e
                ]

                entity_count = len(entity_names)

                summary_text = c["text"][:160].replace("\n", " ").strip()
                word_count = len(c["text"].split())

                meta = {
                    "chunk_id": idx,  # CHANGE: Stable identifier for citations
                    "text": c["text"],
                    "summary": summary_text,  # CHANGE: for reference to lookup in faiss db
                    "start_time": c["start_time"],
                    "end_time": c["end_time"],
                    "chapter_id": chapter,
                    "chapter_number": chapter,  # CHANGE: duplicate for intuitive filtering
                    "scene_id": scene_counter,  # CHANGE: keeps event chunks grouped
                    "token_count": c["token_count"],
                    "word_count": word_count,  # CHANGE: helps score noisy chunks
                    "entities": entity_names ,  # CHANGE: entity-aware retrieval
                    "entity_count": entity_count,
                    "source_file": c.get("source_file"),
                    "file_basename": os.path.basename(c.get("source_file") or "unknown")  # CHANGE: clean citations
                }

                metadatas.append(meta)


            if metadatas:
                logger.info(f"Entities: {metadatas[0]['entities']}")
                logger.info(f"Start Time: {metadatas[0]['start_time']}")
                logger.info(f"End Time: {metadatas[0]['end_time']}")
            logger.info("[META DATA] Complete !! Hurray")

            # --------------------------------------------------------------------
            # CHANGE END
            # --------------------------------------------------------------------

            t4 = time.time()
            vector_db.add_documents(embeddings, metadatas)
            t5 = time.time()

            vector_db_added = True
            mlflow.log_metric("time_vector_db_write_sec", t5 - t4)

            mlflow.log_artifact(vector_db.index_file, artifact_path="faiss")
            mlflow.log_artifact(vector_db.metadata_file, artifact_path="faiss")

            mlflow.log_metric("faiss_index_size_mb", os.path.getsize(vector_db.index_file) / 1e6)
            mlflow.log_metric("faiss_metadata_size_mb", os.path.getsize(vector_db.metadata_file) / 1e6)

        # -----------------------------
        # Plots + DB Snapshot
        # -----------------------------
        metadata_db_path = "audiobook_metadata.db"
        if os.path.exists(metadata_db_path):
            mlflow.log_artifact(metadata_db_path, artifact_path="db_snapshot")

        end_time_total = time.time()
        mlflow.log_metric("total_pipeline_time_sec", end_time_total - start_time_total)
        mlflow.end_run()

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


'''
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
'''
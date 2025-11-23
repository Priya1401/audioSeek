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
class QAService:
    """Service for question answering"""

    def __init__(self, metadata_db: MetadataDBService):
        self.metadata_db = metadata_db

        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-flash-latest')

    def parse_query(self, query: str):
        import re
        q = query.lower()

        if "till chapter" in q:
            m = re.search(r"till chapter (\d+)", q)
            if m:
                return "till_chapter", int(m.group(1))

        if "chapter" in q:
            m = re.search(r"chapter (\d+)", q)
            if m:
                return "chapter", int(m.group(1))

        if "at" in q:
            m = re.search(r'at (\d+):(\d+):(\d+)', q)
            if m:
                h, m2, s = map(int, m.groups())
                return "timestamp", h * 3600 + m2 * 60 + s

        return "general", None

    # ---------------------------
    # METADATA CHUNK RETRIEVAL
    # ---------------------------
    def get_chunks_from_metadata(self, query_type, param, book_id: str):
        if query_type == "chapter":
            return self.metadata_db.get_chunks(book_id, chapter_id=param)[
                "chunks"]

        if query_type == "till_chapter":
            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            target_ids = [c[0] for c in chapters if c[1] <= param]

            results = []
            for cid in target_ids:
                results.extend(
                    self.metadata_db.get_chunks(book_id, cid)["chunks"])
            return results

        if query_type == "timestamp":
            all_chunks = self.metadata_db.get_chunks(book_id)["chunks"]
            return [c for c in all_chunks if c[2] <= param <= c[3]]

        return []


    # ---------------------------
    # LLM PROMPT GENERATORS - Created new function to generate more similar queries to existing query
    # ---------------------------

    def expand_query(self, query: str) -> List[str]:
        """"Generate query variations for better retrieval """

        # Using the same model to generate variations of prompts given by user

        # Skip expansion for very simple queries
        if len(query.split()) <= 3:
            logger.info("Simple Query, skipping expansion!")
            return [query]

        expansion_prompt = f'''Generate 4 alternative phrasings of this question that mean the same thing: "{query}"
        Requirements:
        - Use different verbs
        - Use different contexts if applicable
        - Keep the core meaning
        - Make them search-friendly
        Return ONLY 4 variations, one per line, no numbering.
        '''

        try:
            response = self.llm.generate_content(expansion_prompt)
            variations = [line.strip() for line in response.text.strip().split('\n') if line.strip()]

            # Add original query
            all_queries = [query] + variations[:4]  # Original + 4 variations = 5 total
            return all_queries

        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [query]  # Fallback to original query


    # ---------------------------
    # LLM ANSWER GENERATION
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

        {history_context}

        Context Passages:
        {context}

        Question: {query}

        Answer:"""
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error generating answer: {e}"

    # ---------------------------
    # MAIN QA HANDLER
    # ---------------------------
    def ask_question(self, request: QueryRequest):
        book_id = request.book_id if request.book_id else "default"
        logger.info(f"QA for book_id: {book_id}, query: {request.query}")

        # Session Management
        session_id = request.session_id
        chat_history = []
        if not session_id:
            session_id = self.metadata_db.create_session()
            logger.info(f"Created new session: {session_id}")
        else:
            chat_history = self.metadata_db.get_chat_history(session_id)
            logger.info(f"Retrieved {len(chat_history)} messages for session {session_id}")

        # Start MLflow run
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.start_run(run_name=f"qa_ask_{book_id}")

        mlflow.log_param("book_id", book_id)
        mlflow.log_param("query", request.query)
        mlflow.log_param("top_k", request.top_k)
        mlflow.log_param("session_id", session_id)

        start_time = time.time()

        query_type, param = self.parse_query(request.query)
        mlflow.log_param("query_type", query_type)
        if param is not None:
            mlflow.log_param("query_param", param)

        # ALWAYS TRY VECTOR SEARCH FIRST (it has the data!)
        vector_db = get_vector_db(book_id=book_id)

        expanded_queries = self.expand_query(request.query)
        logger.info(f"Expanded queries : {expanded_queries}")

        all_results = []
        for query in expanded_queries:
            embedding = embedding_model.encode([query])[0].tolist()
            results = vector_db.search(embedding, request.top_k)
            all_results.extend(results)

        # Log embedding dimension once
        sample_embedding = embedding_model.encode(["test"])[0].tolist()
        mlflow.log_metric("embedding_dim", len(sample_embedding))

        # Ensure avoiding same chunks over and over
        seen_chunks = set()
        unique_results = []
        for result in all_results:
            text = result['metadata']['text']
            if text not in seen_chunks:
                seen_chunks.add(text)
                unique_results.append(result)

        results = unique_results[:request.top_k]

        #results = vector_db.search(embedding, request.top_k)

        # If chapter query, filter by chapter_id
        if query_type == "chapter" and param and results:
            logger.info(f"Filtering {len(results)} results for chapter {param}")
            filtered = [r for r in results if
                        r['metadata'].get('chapter_id') == param]
            if filtered:
                results = filtered
                logger.info(
                    f"Found {len(filtered)} results for chapter {param}")
            else:
                logger.warning(
                    f"No chapter {param} results, using all {len(results)} results")

        if results:
            # Vector search worked
            results.sort(key=lambda x: x["score"], reverse=True)
            texts = [r["metadata"]["text"] for r in results]
            citations = [
                f"{r['metadata'].get('start_time')}-{r['metadata'].get('end_time')}"
                for r in results
            ]

            mlflow.log_metric("search_results", len(results))
            mlflow.log_param("search_method", "vector_db")

            if results:
                top = results[0]
                mlflow.log_param("top_result_score", top["score"])
                mlflow.log_dict(
                    {"top_result_metadata": top["metadata"]},
                    "top_result.json"
                )

        elif query_type in ["chapter", "till_chapter", "timestamp"]:
            # Fallback to metadata DB if vector search fails
            logger.info(
                f"Vector search returned no results, trying metadata DB for {query_type}")
            chunks = self.get_chunks_from_metadata(query_type, param, book_id)

            if not chunks:
                mlflow.log_metric("chunks_returned", 0)
                mlflow.log_param("search_method", "metadata_db_empty")
                mlflow.end_run()
                return QueryResponse(answer="No relevant content found",
                                     citations=[], session_id=session_id)

            texts = [c[4] for c in chunks]
            citations = [f"{c[2]}-{c[3]}" for c in chunks]
            mlflow.log_metric("chunks_returned", len(texts))
            mlflow.log_param("search_method", "metadata_db")

        else:
            # No results from vector search and not a special query type
            mlflow.log_metric("search_results", 0)
            mlflow.log_param("search_method", "none")
            mlflow.end_run()
            return QueryResponse(
                answer="No relevant information found.",
                citations=[],
                session_id=session_id
            )

        # Generate answer
        logger.info(f"Generating answer from {len(texts)} text passages")
        answer = self.generate_answer(request.query, texts, chat_history)

        # Save to history
        self.metadata_db.add_chat_message(session_id, "user", request.query)
        self.metadata_db.add_chat_message(session_id, "assistant", answer)

        mlflow.log_metric("answer_length", len(answer or ""))
        mlflow.log_text(answer, "answer.txt")
        mlflow.log_dict({"citations": citations}, "citations.json")

        total_time = time.time() - start_time
        mlflow.log_metric("qa_total_time_sec", total_time)

        mlflow.end_run()

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
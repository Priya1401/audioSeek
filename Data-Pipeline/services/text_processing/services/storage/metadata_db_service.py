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

    def get_all_audiobooks(self):
        """Get all audiobooks from the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT book_id, title, author, duration FROM audiobooks")
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def sync_from_gcs(self, project_id: str, bucket_name: str):
        """Scan GCS bucket for books and populate metadata DB"""
        try:
            from google.cloud import storage
            client = storage.Client(project=project_id)
            bucket = client.bucket(bucket_name)

            # List "directories" in vector-db/
            # GCS doesn't have real directories, so we list blobs with prefix and delimiter
            blobs = bucket.list_blobs(prefix="vector-db/", delimiter="/")
            
            # Force iteration to populate prefixes (directories)
            list(blobs)
            
            prefixes = blobs.prefixes
            logger.info(f"Found book prefixes in GCS: {prefixes}")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for prefix in prefixes:
                # prefix is like "vector-db/book_id/"
                parts = prefix.strip("/").split("/")
                if len(parts) >= 2:
                    book_id = parts[1]
                    title = book_id.replace("_", " ").title()
                    
                    logger.info(f"Discovered book from GCS: {book_id}")
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO audiobooks (book_id, title, author)
                        VALUES (?, ?, ?)
                    """, (book_id, title, "Unknown (GCS)"))

            conn.commit()
            conn.close()
            logger.info("Metadata DB synced with GCS")

        except Exception as e:
            logger.error(f"Failed to sync metadata from GCS: {e}")

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
        # self.llm = genai.GenerativeModel('gemini-flash-latest')
        self.llm = genai.GenerativeModel(
            'gemini-flash-latest',
            generation_config=genai.types.GenerationConfig(
                temperature=0.0
            )
        )
    def parse_query(self, query: str):
        import re
        q = query.lower()

        if "till chapter" in q:
            m = re.search(r"till chapter (\d+)", q)
            if m:
                return "till_chapter", int(m.group(1))

        if "chapter" in q:
            # Extract all numbers to handle "chapter 1 and 2"
            numbers = [int(n) for n in re.findall(r'\d+', q)]
            if numbers:
                return "chapters", numbers

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
        if query_type == "chapters":
            # param is a list of chapter IDs
            all_chunks = []
            for cid in param:
                chunks = self.metadata_db.get_chunks(book_id, chapter_id=cid)
                all_chunks.extend(chunks)
            return all_chunks

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
    def generate_answer(self, query: str, context_results: List[Dict], chat_history: List[Dict[str, str]] = None):
        context = "\n".join(
            [f"Passage {i + 1} (Chapter {r['metadata'].get('chapter_id', 'Unknown')}):\n{r['metadata']['text']}\n"
             for i, r in enumerate(context_results)]
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
        # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        # mlflow.start_run(run_name=f"qa_ask_{book_id}")

        # mlflow.log_param("book_id", book_id)
        # mlflow.log_param("query", request.query)
        # mlflow.log_param("top_k", request.top_k)
        # mlflow.log_param("session_id", session_id)

        start_time = time.time()

        query_type, param = self.parse_query(request.query)
        # mlflow.log_param("query_type", query_type)
        # if param is not None:
        #     mlflow.log_param("query_param", param)

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
        # mlflow.log_metric("embedding_dim", len(sample_embedding))

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
        # If chapter query, filter by chapter_id
        if query_type == "chapters" and param:
            target_chapters = param if isinstance(param, list) else [param]
            logger.info(f"Filtering results for chapters {target_chapters}")
            
            # 1. Filter vector results
            filtered_vector = [r for r in results if
                        r['metadata'].get('chapter_id') in target_chapters]
            
            # 2. Identify missing chapters
            found_chapters = set(r['metadata'].get('chapter_id') for r in filtered_vector)
            missing_chapters = [c for c in target_chapters if c not in found_chapters]
            
            # 3. Fetch missing chapters from Vector DB Metadata (reliable source)
            if missing_chapters:
                logger.info(f"Vector search missed chapters {missing_chapters}, fetching from Vector DB metadata")
                
                for cid in missing_chapters:
                    # vector_db.get_by_chapter returns list of metadata dicts
                    chapter_metas = vector_db.get_by_chapter(cid)
                    
                    for meta in chapter_metas:
                        # meta is already in the correct format: {text, start_time, ...}
                        filtered_vector.append({"score": 1.0, "metadata": meta})

            results = filtered_vector
            logger.info(f"Final results count for chapters {target_chapters}: {len(results)}")

        if results:
            # Vector search worked
            results.sort(key=lambda x: x["score"], reverse=True)
            texts = [r["metadata"]["text"] for r in results]
            citations = [
                f"{r['metadata'].get('start_time')}-{r['metadata'].get('end_time')}"
                for r in results
            ]

            # mlflow.log_metric("search_results", len(results))
            # mlflow.log_param("search_method", "vector_db")

            if results:
                top = results[0]
                # mlflow.log_param("top_result_score", top["score"])
                # mlflow.log_dict(
                #     {"top_result_metadata": top["metadata"]},
                #     "top_result.json"
                # )

        elif query_type in ["chapter", "till_chapter", "timestamp"]:
            # Fallback to metadata DB if vector search fails
            logger.info(
                f"Vector search returned no results, trying metadata DB for {query_type}")
            chunks = self.get_chunks_from_metadata(query_type, param, book_id)

            if not chunks:
                # mlflow.log_metric("chunks_returned", 0)
                # mlflow.log_param("search_method", "metadata_db_empty")
                # mlflow.end_run()
                return QueryResponse(answer="No relevant content found",
                                     citations=[], session_id=session_id)

            texts = [c[4] for c in chunks]
            citations = [f"{c[2]}-{c[3]}" for c in chunks]
            # mlflow.log_metric("chunks_returned", len(texts))
            # mlflow.log_param("search_method", "metadata_db")

        else:
            # No results from vector search and not a special query type
            # mlflow.log_metric("search_results", 0)
            # mlflow.log_param("search_method", "none")
            # mlflow.end_run()
            return QueryResponse(
                answer="No relevant information found.",
                citations=[],
                session_id=session_id
            )

        # Generate answer
        logger.info(f"Generating answer from {len(results)} text passages")
        answer = self.generate_answer(request.query, results, chat_history)

        # Save to history
        self.metadata_db.add_chat_message(session_id, "user", request.query)
        self.metadata_db.add_chat_message(session_id, "assistant", answer)

        # mlflow.log_metric("answer_length", len(answer or ""))
        # mlflow.log_text(answer, "answer.txt")
        # mlflow.log_dict({"citations": citations}, "citations.json")

        total_time = time.time() - start_time
        # mlflow.log_metric("qa_total_time_sec", total_time)

        # mlflow.end_run()

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
        # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # Safety: End any dangling run on this thread
        # if mlflow.active_run():
        #     logger.warning(f"Found active run {mlflow.active_run().info.run_id}, ending it.")
        #     mlflow.end_run()

        # mlflow.start_run(run_name=f"process_full_{book_id}")

        # mlflow.log_param("book_id", book_id)
        # mlflow.log_param("target_tokens", request.target_tokens)
        # mlflow.log_param("overlap_tokens", request.overlap_tokens)
        # mlflow.log_param("add_to_vector_db", request.add_to_vector_db)

        start_time_total = time.time()

        # -----------------------------
        # 2. Log raw file list (not content)
        # -----------------------------
        if request.folder_path and os.path.exists(request.folder_path):
            file_list = sorted([
                f for f in os.listdir(request.folder_path)
                if f.endswith(".txt")
            ])
            # mlflow.log_dict(
            #     {"folder": request.folder_path, "files": file_list},
            #     artifact_file="raw_input_file_list.json"
            # )

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
        # mlflow.log_metric("num_chunks", len(chunk_response.chunks))
        # mlflow.log_metric("num_chapters", len(chunk_response.chapters))
        # mlflow.log_metric("num_entities", len(chunk_response.entities))
        # mlflow.log_metric("time_chunking_sec", t1 - t0)

        # Artifact: chunks
        chunk_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(chunk_artifact.name, "w") as f:
            json.dump(chunk_response.chunks, f, indent=2)
        # mlflow.log_artifact(chunk_artifact.name, artifact_path="chunks")

        # Artifact: chapters
        chapter_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(chapter_artifact.name, "w") as f:
            json.dump(chunk_response.chapters, f, indent=2)
        # mlflow.log_artifact(chapter_artifact.name, artifact_path="chapters")

        # Artifact: entities
        entity_artifact = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(entity_artifact.name, "w") as f:
            json.dump(chunk_response.entities, f, indent=2)
        # mlflow.log_artifact(entity_artifact.name, artifact_path="entities")

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

        # mlflow.log_metric("embedding_count", len(embeddings))
        # mlflow.log_metric("time_embedding_sec", t3 - t2)
        # mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

        # embeddings.npy artifact
        embed_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(embed_file.name, np.array(embeddings))
        # mlflow.log_artifact(embed_file.name, artifact_path="embeddings")

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
            # mlflow.log_metric("time_vector_db_write_sec", t5 - t4)

            # Log FAISS artifacts
            # mlflow.log_artifact(vector_db.index_file, artifact_path="faiss")
            # mlflow.log_artifact(vector_db.metadata_file, artifact_path="faiss")

            # mlflow.log_metric("faiss_index_size_mb", os.path.getsize(vector_db.index_file) / 1e6)
            # mlflow.log_metric("faiss_metadata_size_mb", os.path.getsize(vector_db.metadata_file) / 1e6)

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
        # mlflow.log_artifact(token_plot.name, artifact_path="plots")
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
            # mlflow.log_artifact(chapter_plot.name, artifact_path="plots")
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
            # mlflow.log_artifact(ent_plot.name, artifact_path="plots")
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
                # mlflow.log_artifact(sim_plot.name, artifact_path="plots")
                plt.close()
        except Exception as e:
            logger.error(f"Error creating similarity heatmap: {e}")

        # -----------------------------
        # 7. Log DB snapshot
        # -----------------------------
        metadata_db_path = "audiobook_metadata.db"
        if os.path.exists(metadata_db_path):
            pass
            # mlflow.log_artifact(metadata_db_path, artifact_path="db_snapshot")

        # -----------------------------
        # 8. Finish MLflow Run
        # -----------------------------
        end_time_total = time.time()
        # mlflow.log_metric("total_pipeline_time_sec", end_time_total - start_time_total)
        # mlflow.end_run()

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

        # mlflow.end_run()
        return response

# Global instance
metadata_db = MetadataDBService()

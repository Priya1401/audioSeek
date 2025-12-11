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
from services.storage.vector_db_service import get_vector_db
from services.nlp.chunking_service import ChunkingService

from core.model_loader import get_embedding_model

embedding_model = get_embedding_model()

logger = logging.getLogger(__name__)

# Load embedding model
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class MetadataDBService:
    """Book-aware metadata database operations"""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.getenv("METADATA_DB_PATH", "audiobook_metadata.db")
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
        """Get all audiobooks from the database with chapter counts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT 
            b.book_id, 
            b.title, 
            b.author, 
            b.duration,
            COUNT(c.id) as chapter_count
        FROM audiobooks b
        LEFT JOIN chapters c ON b.book_id = c.book_id
        GROUP BY b.book_id
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_system_stats(self):
        """Get overall system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM audiobooks")
        stats["total_books"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chapters")
        stats["total_chapters"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunks")
        stats["total_chunks"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM entities")
        stats["total_entities"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sessions")
        stats["total_sessions"] = cursor.fetchone()[0]

        conn.close()
        return stats

    def get_detailed_book_stats(self):
        """Get detailed stats per book (chapters, chunks count)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT 
            b.book_id, 
            b.title, 
            COUNT(DISTINCT c.id) as chapter_count,
            COUNT(DISTINCT ch.id) as chunk_count
        FROM audiobooks b
        LEFT JOIN chapters c ON b.book_id = c.book_id
        LEFT JOIN chunks ch ON b.book_id = ch.book_id
        GROUP BY b.book_id
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            r = dict(row)

            
            results.append(r)

        return results

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

        except Exception as e:
            logger.error(f"Failed to list GCS prefixes: {e}")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            for prefix in prefixes:
                # prefix is like "vector-db/book_id/"
                parts = prefix.strip("/").split("/")
                if len(parts) >= 2:
                    book_id = parts[1]
                    title = book_id.replace("_", " ").title()
                    logger.info(f"Discovered book from GCS: {book_id}")

                    # 1. Ensure Book Exists
                    cursor.execute("""
                        INSERT OR IGNORE INTO audiobooks (book_id, title, author)
                        VALUES (?, ?, ?)
                    """, (book_id, title, "Unknown (GCS)"))
                    
                    # 2. Try to fetch metadata.json to restore chunks/chapters
                    try:
                        blob_path = f"{prefix}metadata.json"
                        blob = bucket.blob(blob_path)
                        
                        if blob.exists():
                            logger.info(f"Downloading metadata for {book_id}...")
                            metadata_content = blob.download_as_text()
                            chunk_data = json.loads(metadata_content)
                            
                            # chunk_data is a list of dicts (from FAISSVectorDB)
                            # [{ "text":..., "chapter_id":..., "start_time":... }]

                            # A. Recover Chapters
                            # We can't recover titles/summaries perfectly, but we can recreate IDs
                            chapter_ids = sorted(list(set(
                                c.get('chapter_id') for c in chunk_data 
                                if c.get('chapter_id') is not None
                            )))
                            
                            for cid in chapter_ids:
                                # Check if chapter exists
                                cursor.execute("SELECT 1 FROM chapters WHERE book_id=? AND chapter_number=?", (book_id, cid))
                                if not cursor.fetchone():
                                    cursor.execute("""
                                        INSERT INTO chapters (book_id, chapter_number, title, start_time, end_time)
                                        VALUES (?, ?, ?, 0, 0)
                                    """, (book_id, cid, f"Chapter {cid}"))

                            # B. Recover Chunks
                            # Check if we already have chunks to avoid dupes
                            cursor.execute("SELECT COUNT(*) FROM chunks WHERE book_id=?", (book_id,))
                            count = cursor.fetchone()[0]
                            
                            if count == 0:
                                logger.info(f"Restoring {len(chunk_data)} chunks for {book_id}...")
                                chunk_tuples = []
                                for c in chunk_data:
                                    # Lookup internal chapter ID (which is just rowid, but our schema uses integer ID)
                                    # Wait, schema uses `id` integer primary key for chapters.
                                    # But we inserted with `chapter_number`.
                                    # We need to map chapter_number -> chapter_id (rowid)
                                    
                                    c_num = c.get("chapter_id")
                                    internal_chap_id = None
                                    if c_num is not None:
                                        cursor.execute("SELECT id FROM chapters WHERE book_id=? AND chapter_number=?", (book_id, c_num))
                                        row = cursor.fetchone()
                                        if row:
                                            internal_chap_id = row[0]
                                    
                                    chunk_tuples.append((
                                        book_id,
                                        internal_chap_id,
                                        c.get("start_time", 0),
                                        c.get("end_time", 0),
                                        c.get("text", ""),
                                        c.get("token_count", 0),
                                        c.get("source_file", "")
                                    ))
                                
                                cursor.executemany("""
                                    INSERT INTO chunks (book_id, chapter_id, start_time, end_time, text, token_count, source_file)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, chunk_tuples)
                                logger.info(f"Restored {len(chunk_tuples)} chunks.")
                            else:
                                logger.info(f"Chunks already exist for {book_id}, skipping restore.")

                    except Exception as meta_err:
                        logger.warning(f"Could not restore metadata for {book_id}: {meta_err}")

            conn.commit()
            logger.info("Metadata DB sync complete.")
            
        except Exception as e:
            logger.error(f"Sync loop failed: {e}")
        finally:
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
        # self.llm = genai.GenerativeModel('gemini-flash-latest')
        self.llm = genai.GenerativeModel(
            'gemini-flash-latest',
            generation_config=genai.types.GenerationConfig(
                temperature=0.0
            )
        )

    def parse_query(self, query: str):
        import re
        q = query.lower().strip()

        # Timestamp range: "between 1:10:00 and 1:12:00"
        m = re.search(r'between (\d+):(\d+):(\d+) and (\d+):(\d+):(\d+)', q)
        if m:
            t1 = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            t2 = int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
            return ("timestamp_range", (t1, t2))

        # Single timestamp: "at 1:12:30"
        m = re.search(r'at (\d+):(\d+):(\d+)', q)
        if m:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            return ("timestamp", t)

        # Spoiler-safe instruction
        if "no spoilers" in q or "spoiler" in q or "don't spoil" in q:
            return ("spoiler_safe", None)

        if "till chapter" in q or "up to chapter" in q:
            m = re.search(r'chapter (\d+)', q)
            if m:
                return ("till_chapter", int(m.group(1)))

        if "chapter" in q:
            nums = [int(x) for x in re.findall(r'\d+', q)]
            if nums:
                return ("chapters", nums)

        return ("general", None)



    # ---------------------------
    # METADATA CHUNK RETRIEVAL
    # ---------------------------
    def get_chunks_from_metadata(self, query_type, param, book_id: str):
        if query_type == "chapters":
            # param is a list of chapter IDs
            all_chunks = []
            for cid in param:
                chunks_dict = self.metadata_db.get_chunks(book_id, chapter_id=cid)
                all_chunks.extend(chunks_dict["chunks"])
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
    def generate_answer(
            self,
            query: str,
            context_results: List[Dict],
            chat_history: List[Dict[str, str]] = None
    ):
        # Build plain text context from retrieved passages
        context = "\n".join(
            [
                f"Passage {i + 1} (Chapter {r['metadata'].get('chapter_id', 'Unknown')}, "
                f"{r['metadata'].get('start_time', 0)}–{r['metadata'].get('end_time', 0)} sec):\n"
                f"{r['metadata']['text']}\n"
                for i, r in enumerate(context_results)
            ]
        )

        history_context = ""
        if chat_history:
            history_context = (
                    "\nPrevious Chat History:\n"
                    + "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history]
            )
                    + "\n"
            )

        query_lower = query.lower()

        # --------------------------------------------------------
        # Detect if this is a "summary style" question
        # (chapter summary or "until chapter" recap)
        # --------------------------------------------------------
        is_chapter_summary = any(
            phrase in query_lower
            for phrase in [
                "what happens in chapter",
                "what happened in chapter",
                "summary of chapter",
                "summarize chapter",
                "can you summarize chapter",
                "summarise chapter",
                "what happens until chapter",
                "what happened until chapter",
                "summarize until chapter",
                "summarise until chapter",
                "until chapter",
                "till chapter",
            ]
        )

        if is_chapter_summary:
            # More narrative, structured summary style
            prompt = f"""
You are summarizing an audiobook for a reader.

GOAL:
- Give a clear, engaging summary of the events described in the context.
- Explain what happens in order, highlighting key events and character actions.
- If the user asks "until Chapter X" or "up to Chapter X", treat the context as everything that has happened in the story up to that point.

WRITING STYLE:
- Start with 1–2 sentences that briefly state the main situation or turning point.
- Then use 3–6 concise bullet points to describe the major events in chronological order.
- Paraphrase; do NOT just repeat raw lines from the context.
- Focus on the most important plot points, not tiny details.
- Do not invent events that are not clearly supported by the context.

{history_context}

CONTEXT:
{context}

QUESTION:
{query}

Now write a coherent, spoiler-aware summary:
"""
        else:
            # General QA style
            prompt = f"""You are a helpful assistant answering questions about an audiobook.

INSTRUCTIONS:
1. Answer using the information from the provided context passages.
2. You may make reasonable inferences based on character relationships and interactions described.
3. If characters support each other, speak intimately, or consistently help one another, you can infer close friendship.
4. Be helpful and direct - don't be overly cautious.
5. If information truly isn't in the context, say so clearly.
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

    def generate_when_answer(
            self,
            query: str,
            context_results: List[Dict],
            chat_history: List[Dict[str, str]] = None
    ):
        """
        Specialized helper for "WHEN" questions.
        Given multiple candidate passages (with chapter + timestamps),
        ask the LLM to decide which one actually matches the user's question
        (e.g., first time, finally, etc.) and return structured info.
        """
        context = ""
        for i, r in enumerate(context_results):
            m = r["metadata"]
            context += (
                f"Passage {i + 1}:\n"
                f"- Chapter: {m.get('chapter_id')}\n"
                f"- Start: {m.get('start_time')} sec\n"
                f"- End: {m.get('end_time')} sec\n"
                f"- Text: {m['text']}\n\n"
            )

        history_context = ""
        if chat_history:
            history_context = (
                    "\nPrevious Chat History:\n"
                    + "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in chat_history
            )
                    + "\n"
            )

        prompt = f"""
You answer questions about WHEN something happens in an audiobook.

You are given multiple passages that may mention the event in different ways:
- earlier hints or attempts
- the actual moment the event happens
- references after it has already happened

Your job:
1. Read ALL passages carefully.
2. Figure out **which passage best matches what the user is asking**:
   - If the question says "first" or "first time", choose the earliest passage where the event truly happens.
   - If the question says "finally", "at last", "eventually", choose the passage where the event is actually completed after earlier attempts.
   - If the wording is neutral ("When does X happen?"), choose the most natural/explicit occurrence.
3. Output a STRICT JSON object with:
   - "chapter_id": the chapter number as an integer
   - "start_time": the best estimated start time in seconds (from the given passages)
   - "end_time": the matching end time in seconds
   - "reason": a one-sentence explanation of why this passage is the best match.

Do NOT invent timestamps that are not in the passages.
Only choose chapter_id and times that are present in the input.

{history_context}

CONTEXT:
{context}

QUESTION:
{query}

Now respond ONLY with JSON, no extra text.
"""

        try:
            resp = self.llm.generate_content(prompt)
            raw = resp.text.strip()

            # Very light cleanup in case the model wraps JSON in code fences
            if raw.startswith("```"):
                raw = raw.strip("`")
                # remove possible "json" tag
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            import json
            parsed = json.loads(raw)
            return parsed
        except Exception as e:
            logger.error(f"generate_when_answer error: {e}")
            return None

    def generate_spoiler_safe_answer(self, query, passages, chat_history=None):
        context = ""
        for i, r in enumerate(passages):
            meta = r["metadata"]
            context += (
                f"Passage {i + 1}:\n"
                f"- Chapter: {meta.get('chapter_id')}\n"
                f"- Start: {meta.get('start_time')} sec\n"
                f"- End: {meta.get('end_time')} sec\n"
                f"- Text: {meta['text']}\n\n"
            )

        prompt = f"""
    You are answering a timestamp-based question about an audiobook.

    SPOILER RULES:
    - Only describe what is happening within the provided passages.
    - Do NOT reveal information from later chapters or later timestamps.
    - If asked about anything outside the timestamps, politely refuse.
    - If the user wants story details, only reference what is inside the context.

    Context:
    {context}

    Question: {query}

    Provide a spoiler-safe answer:
    """

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"LLM error: {e}"


    # ---------------------------
    # MAIN QA HANDLER
    # ---------------------------
    def ask_question(self, request: QueryRequest):
        book_id = request.book_id if request.book_id else "default"
        logger.info(f"QA for book_id: {book_id}, query: {request.query}")

        chapter_read = None
        completed_timestamp = None

        if request.until_chapter:
            chapter_read = request.until_chapter

        if request.until_time_seconds:
            completed_timestamp = request.until_time_seconds

        # -------------------------
        # Session Handling
        # -------------------------
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
        query_lower = request.query.lower()

        import re

        # ------------------------------------------------------------
        # helper for pretty timestamp formatting
        # ------------------------------------------------------------
        def fmt(sec: float) -> str:
            hh = int(sec // 3600)
            mm = int((sec % 3600) // 60)
            ss = int(sec % 60)
            return f"{hh:02d}:{mm:02d}:{ss:02d}"

        # ============================================================
        # "UNTIL CHAPTER N AT TIME" HANDLER
        # e.g. "What happens until Chapter 4 at around 00:07:46?"
        #      "What happens till Chapter 4 at 07:46?"
        # Logic:
        #   - Take ALL chunks from chapters 1..(N-1)
        #   - Take chunks from Chapter N whose start_time <= given timestamp
        #   - Summarize all of that without going beyond that point
        # ============================================================
        m_until = re.search(
            r'(?:until|till)\s+chapter\s+(\d+).*?(?:at|around)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?',
            query_lower
        )
        if m_until:
            target_chapter = int(m_until.group(1))
            # Parse timestamp (supports mm:ss or hh:mm:ss)
            if m_until.group(4):  # hh:mm:ss
                h = int(m_until.group(2))
                m = int(m_until.group(3))
                s = int(m_until.group(4))
            else:                 # mm:ss
                h = 0
                m = int(m_until.group(2))
                s = int(m_until.group(3))
            target_ts = h * 3600 + m * 60 + s

            logger.info(
                f"[UNTIL] Up to Chapter {target_chapter} at {fmt(target_ts)} "
                f"for book_id={book_id}"
            )

            # 1) Get chapters, build mappings:
            # chapters rows: (id_pk, chapter_number, title, start_time, end_time, summary)
            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            chap_pk_to_num = {row[0]: row[1] for row in chapters}  # id_pk -> chapter_number
            chap_num_to_pk = {row[1]: row[0] for row in chapters}  # chapter_number -> id_pk

            if target_chapter not in chap_num_to_pk:
                return QueryResponse(
                    answer=f"Chapter {target_chapter} not found.",
                    citations=[],
                    session_id=session_id
                )

            # 2) Get all chunks, filter by chapter_number + timestamp
            all_chunks = self.metadata_db.get_chunks(book_id)["chunks"]
            # chunks rows: (id, chapter_id_fk, start_time, end_time, text, token_count)
            context_chunks = []
            for c in all_chunks:
                chap_pk = c[1]
                chap_num = chap_pk_to_num.get(chap_pk)
                if chap_num is None:
                    continue

                # earlier chapters: include fully
                if chap_num < target_chapter:
                    context_chunks.append(c)
                # target chapter: only chunks whose start_time <= target_ts
                elif chap_num == target_chapter and c[2] <= target_ts:
                    context_chunks.append(c)

            if not context_chunks:
                return QueryResponse(
                    answer=(
                        f"No content found up to Chapter {target_chapter} "
                        f"at {fmt(target_ts)}."
                    ),
                    citations=[],
                    session_id=session_id
                )

            # 3) Convert chunks into context_results for LLM
            context_results = []
            for c in context_chunks:
                chap_pk = c[1]
                chap_num = chap_pk_to_num.get(chap_pk)
                context_results.append({
                    "metadata": {
                        "chapter_id": chap_num,
                        "start_time": c[2],
                        "end_time": c[3],
                        "text": c[4],
                        "token_count": c[5],
                    }
                })

            # 4) Ask LLM to summarize everything up to that point
            prefix = (
                f"**Summary up to Chapter {target_chapter} at {fmt(target_ts)}:**\n\n"
            )
            llm_answer = self.generate_answer(
                request.query,
                context_results,
                chat_history
            )
            final_answer = prefix + llm_answer

            citations = [f"{c[2]}-{c[3]}" for c in context_chunks]

            # Save chat history
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(session_id, "assistant", final_answer)

            return QueryResponse(
                answer=final_answer,
                citations=citations,
                session_id=session_id
            )

        # ============================================================
        # "UNTIL CHAPTER N" (NO TIMESTAMP)
        # e.g. "Can you summarize what happens until Chapter 3?"
        #      "Give me a recap till Chapter 5"
        # Logic:
        #   - Take ALL chunks from chapters 1..N
        #   - Summarize them as "story so far"
        # ============================================================
        m_until_simple = re.search(
            r'(?:until|till)\s+chapter\s+(\d+)',
            query_lower
        )
        if m_until_simple:
            target_chapter = int(m_until_simple.group(1))
            logger.info(
                f"[UNTIL SIMPLE] Up to Chapter {target_chapter} for book_id={book_id}"
            )

            # 1) Get chapter metadata: (id, chapter_number, title, start_time, end_time, summary)
            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            if not chapters:
                return QueryResponse(
                    answer="No chapter metadata found for this book.",
                    citations=[],
                    session_id=session_id
                )

            # Map primary-key id -> chapter_number
            chap_pk_to_num = {row[0]: row[1] for row in chapters}

            # 2) Get all chunks, then filter by chapter_number <= target_chapter
            all_chunks = self.metadata_db.get_chunks(book_id)["chunks"]
            context_chunks = []
            for c in all_chunks:
                chap_pk = c[1]          # chapter_id column in chunks table
                chap_num = chap_pk_to_num.get(chap_pk)
                if chap_num is None:
                    continue
                if chap_num <= target_chapter:
                    context_chunks.append(c)

            if not context_chunks:
                return QueryResponse(
                    answer=f"No content found up to Chapter {target_chapter}.",
                    citations=[],
                    session_id=session_id
                )

            # 3) Build context_results structure for generate_answer
            context_results = []
            for c in context_chunks:
                chap_num = chap_pk_to_num.get(c[1])
                context_results.append({
                    "metadata": {
                        "chapter_id": chap_num,
                        "start_time": c[2],
                        "end_time": c[3],
                        "text": c[4],
                        "token_count": c[5],
                    }
                })

            # 4) Let LLM summarize everything so far
            prefix = f"**Summary up to Chapter {target_chapter}:**\n\n"
            llm_answer = self.generate_answer(
                request.query,
                context_results,
                chat_history
            )
            final_answer = prefix + llm_answer

            citations = [f"{c[2]}-{c[3]}" for c in context_chunks]

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(session_id, "assistant", final_answer)

            return QueryResponse(
                answer=final_answer,
                citations=citations,
                session_id=session_id
            )


        # ============================================================
        # UNIVERSAL TIMESTAMP + CHAPTER QUERY HANDLER  (Primary)
        # Supports:
        #  - "Chapter 3 at 04:10"
        #  - "What happens in Chapter 3 in 00:04:10?"
        #  - "What happens around 00:04:10 in Chapter 3?"
        # ============================================================

        # Case 1: "chapter 3 ... 00:04:10"
        m1 = re.search(
            r'chapter\s+(\d+).*?(\d{1,2}):(\d{2})(?::(\d{2}))?',
            query_lower
        )

        # Case 2: "00:04:10 ... chapter 3"
        m2 = re.search(
            r'(\d{1,2}):(\d{2})(?::(\d{2}))?.*?chapter\s+(\d+)',
            query_lower
        )

        chapter_number = None
        timestamp_sec = None

        if m1:
            # Example: "What happens in Chapter 3 in 00:04:10?"
            chapter_number = int(m1.group(1))
            if m1.group(4):  # hh:mm:ss
                h = int(m1.group(2))
                m = int(m1.group(3))
                s = int(m1.group(4))
            else:            # mm:ss
                h = 0
                m = int(m1.group(2))
                s = int(m1.group(3))
            timestamp_sec = h * 3600 + m * 60 + s

        elif m2:
            # Example: "What happens at 00:04:10 in Chapter 3?"
            if m2.group(3):  # hh:mm:ss
                h = int(m2.group(1))
                m = int(m2.group(2))
                s = int(m2.group(3))
            else:            # mm:ss
                h = 0
                m = int(m2.group(1))
                s = int(m2.group(2))
            timestamp_sec = h * 3600 + m * 60 + s
            chapter_number = int(m2.group(4))

        if chapter_number is not None and timestamp_sec is not None:
            logger.info(f"[Timestamp+Chapter] chapter_number={chapter_number}, t={timestamp_sec}s")

            # Map chapter_number -> DB primary key
            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            # chapters: (id_pk, chapter_number, title, start_time, end_time, summary)
            num_to_pk = {c[1]: c[0] for c in chapters}

            if chapter_number not in num_to_pk:
                return QueryResponse(
                    answer=f"Chapter {chapter_number} not found.",
                    citations=[],
                    session_id=session_id
                )

            chapter_db_id = num_to_pk[chapter_number]

            chapter_chunks = self.metadata_db.get_chunks(
                book_id, chapter_id=chapter_db_id
            )["chunks"]
            chapter_chunks = sorted(chapter_chunks, key=lambda x: x[2])

            match = [c for c in chapter_chunks if c[2] <= timestamp_sec <= c[3]]

            if not match:
                return QueryResponse(
                    answer=f"No event found around {fmt(timestamp_sec)} in Chapter {chapter_number}.",
                    citations=[],
                    session_id=session_id
                )

            chunk = match[0]
            start, end, text = chunk[2], chunk[3], chunk[4]

            prefix = f"**In Chapter {chapter_number}, around {fmt(start)}–{fmt(end)}:**\n\n"

            llm_answer = self.generate_answer(
                request.query,
                [{"metadata": {"text": text}}],
                chat_history
            )

            final_answer = prefix + llm_answer

            # Save chat history
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(session_id, "assistant", final_answer)

            return QueryResponse(
                answer=final_answer,
                citations=[f"{start}-{end}"],
                session_id=session_id
            )

        # ============================================================
        # SINGLE TIMESTAMP QUERY
        # ============================================================
        if query_type == "timestamp":
            t = param
            logger.info(f"[Timestamp Only] t={t}s")

            chunks = self.metadata_db.get_chunks(book_id)["chunks"]
            chunks = sorted(chunks, key=lambda x: x[2])

            matched = [
                {
                    "score": 1.0,
                    "metadata": {
                        "chapter_id": c[1],
                        "start_time": c[2],
                        "end_time": c[3],
                        "text": c[4],
                        "token_count": c[5]
                    }
                }
                for c in chunks if c[2] <= t <= c[3]
            ]

            if not matched:
                return QueryResponse(
                    answer=f"No content found around {fmt(t)}.",
                    citations=[],
                    session_id=session_id
                )

            answer = self.generate_spoiler_safe_answer(request.query, matched, chat_history)
            citations = [f"{m['metadata']['start_time']}-{m['metadata']['end_time']}" for m in matched]

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(session_id, "assistant", answer)

            return QueryResponse(answer=answer, citations=citations, session_id=session_id)

        # ============================================================
        # TIMESTAMP RANGE QUERY
        # ============================================================
        if query_type == "timestamp_range":
            t1, t2 = param
            logger.info(f"Handling timestamp range {t1}-{t2}")

            chunks = self.metadata_db.get_chunks(book_id)["chunks"]

            matched = [
                {
                    "score": 1.0,
                    "metadata": {
                        "chapter_id": c[1],
                        "start_time": c[2],
                        "end_time": c[3],
                        "text": c[4],
                        "token_count": c[5],
                    }
                }
                for c in chunks if not (c[3] < t1 or c[2] > t2)
            ]

            if not matched:
                return QueryResponse(
                    answer="No content within that timestamp range.",
                    citations=[],
                    session_id=session_id
                )

            answer = self.generate_spoiler_safe_answer(request.query, matched, chat_history)
            citations = [f"{m['metadata']['start_time']}-{m['metadata']['end_time']}" for m in matched]

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(session_id, "assistant", answer)

            return QueryResponse(answer=answer, citations=citations, session_id=session_id)

        # ============================================================
        # VECTOR SEARCH (Normal Flow)
        # ============================================================

        # Skip query expansion for pure timestamp/timestamp_range (already handled),
        # but still allow it for normal / chapter-only questions
        if query_type in ["timestamp", "timestamp_range"]:
            expanded_queries = [request.query]
        else:
            expanded_queries = self.expand_query(request.query)

        vector_db = get_vector_db(book_id=book_id)
        logger.info(f"Expanded queries : {expanded_queries}")

        all_results = []
        for q in expanded_queries:
            embedding = embedding_model.encode(
                [q],
                normalize_embeddings=True
            )[0].tolist()
            results = vector_db.search(embedding, request.top_k, chapter_read, completed_timestamp)
            all_results.extend(results)

        # Deduplicate by text
        seen = set()
        unique_results = []
        for r in all_results:
            if r['metadata']['text'] not in seen:
                seen.add(r['metadata']['text'])
                unique_results.append(r)

        results = unique_results[:request.top_k]

        if not results:
            return QueryResponse(
                answer="No relevant information found.",
                citations=[],
                session_id=session_id
            )

        # Sort key: by chapter_id, then start_time
        def chapter_key(r):
            cid = r['metadata'].get('chapter_id')
            if cid is None:
                cid = 10 ** 9
            return (cid, r['metadata']['start_time'])

        # Detect natural-language "when" questions
        is_timestamp_question = any(
            kw in query_lower
            for kw in ["when", "what time", "at what time", "timestamp", "point in the book"]
        )

        timestamp_prefix = ""
        results_for_citation = results

        # Decide context to give to the LLM
        if is_timestamp_question:
            # Let the LLM choose the best passage across all candidates
            when_info = self.generate_when_answer(
                request.query,
                results,
                chat_history
            )

            chosen = None
            if when_info:
                target_chapter = when_info.get("chapter_id")
                target_start = when_info.get("start_time")
                target_end = when_info.get("end_time")

                # Find the matching chunk from results metadata
                for r in results:
                    m = r["metadata"]
                    if (
                            m.get("chapter_id") == target_chapter
                            and abs(m["start_time"] - float(target_start)) < 1e-3
                            and abs(m["end_time"] - float(target_end)) < 1e-3
                    ):
                        chosen = r
                        break

            # Fallback: if parsing or matching failed, just use the earliest chunk
            if chosen is None:
                chosen = min(results, key=chapter_key)

            results_for_llm = [chosen]
            results_for_citation = [chosen]

            meta = chosen["metadata"]
            timestamp_prefix = (
                f"**Timestamp:** This occurs in **Chapter {meta.get('chapter_id')}**, "
                f"around **{fmt(meta['start_time'])}–{fmt(meta['end_time'])}**.\n\n"
            )
        else:
            # For general questions, use all retrieved chunks
            results_for_llm = results

        # -----------------------------
        # Ask the LLM for the answer
        # -----------------------------
        llm_answer = self.generate_answer(request.query, results_for_llm, chat_history)

        # -----------------------------
        # For non-timestamp questions, align citations with chapter LLM mentions
        # -----------------------------
        if not is_timestamp_question:
            chapter_from_llm = None
            m_ch = re.search(r'[Cc]hapter\s+(\d+)', llm_answer)
            if m_ch:
                try:
                    chapter_from_llm = int(m_ch.group(1))
                    logger.info(f"LLM explicitly mentioned Chapter {chapter_from_llm}")
                except ValueError:
                    chapter_from_llm = None

            if chapter_from_llm is not None:
                filtered = [
                    r for r in results_for_citation
                    if r['metadata'].get('chapter_id') == chapter_from_llm
                ]
                if filtered:
                    logger.info(
                        f"Restricting citations to {len(filtered)} chunks from Chapter {chapter_from_llm}"
                    )
                    results_for_citation = filtered

            # Apply explicit chapter filter from query like "in chapter 3"
            if query_type == "chapters" and param:
                target = param if isinstance(param, list) else [param]
                results_for_citation = [
                    r for r in results_for_citation
                    if r['metadata'].get('chapter_id') in target
                ]

            if not results_for_citation:
                return QueryResponse(
                    answer="No relevant information found.",
                    citations=[],
                    session_id=session_id
                )

        # -----------------------------
        # Final answer + citations
        # -----------------------------
        final_answer = timestamp_prefix + llm_answer

        citations = [
            f"{r['metadata']['start_time']}-{r['metadata']['end_time']}"
            for r in results_for_citation
        ]

        # Save chat history
        self.metadata_db.add_chat_message(session_id, "user", request.query)
        self.metadata_db.add_chat_message(session_id, "assistant", final_answer)

        return QueryResponse(
            answer=final_answer,
            citations=citations,
            session_id=session_id
        )


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

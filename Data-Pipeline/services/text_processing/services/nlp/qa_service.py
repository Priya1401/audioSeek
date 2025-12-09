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
import mlflow

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
from services.storage.metadata_db_service import MetadataDBService

from core.model_loader import get_embedding_model

logger = logging.getLogger(__name__)

# Shared embedding model
embedding_model = get_embedding_model()


class QAService:
    """Service for question answering (merged full logic + existing pipeline)."""

    def __init__(self, metadata_db: MetadataDBService):
        self.metadata_db = metadata_db

        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        self.llm = genai.GenerativeModel(
            'gemini-flash-latest',
            generation_config=genai.types.GenerationConfig(
                temperature=0.0
            )
        )

    # ------------------------------------------------------------
    # QUERY PARSING
    # ------------------------------------------------------------
    def parse_query(self, query: str):
        import re
        q = query.lower().strip()

        # Timestamp range: "between 1:10:00 and 1:12:00"
        m = re.search(r'between (\d+):(\d+):(\d+) and (\d+):(\d+):(\d+)', q)
        if m:
            t1 = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            t2 = int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
            return "timestamp_range", (t1, t2)

        # Single timestamp: "at 1:12:30"
        m = re.search(r'at (\d+):(\d+):(\d+)', q)
        if m:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            return "timestamp", t

        # Spoiler-safe instruction
        if "no spoilers" in q or "spoiler" in q or "don't spoil" in q:
            return "spoiler_safe", None

        if "till chapter" in q or "until chapter" in q or "up to chapter" in q:
            m = re.search(r'chapter (\d+)', q)
            if m:
                return "till_chapter", int(m.group(1))

        if "chapter" in q:
            nums = [int(x) for x in re.findall(r'\d+', q)]
            if nums:
                return "chapters", nums

        return "general", None

    # ------------------------------------------------------------
    # METADATA-BASED CHUNK RETRIEVAL
    # ------------------------------------------------------------
    def get_chunks_from_metadata(self, query_type, param, book_id: str):
        """Use sqlite metadata DB to fetch chunks for special query types."""
        if query_type == "chapters":
            # param is a list of chapter numbers
            all_chunks = []
            for cid in param:
                chunks_dict = self.metadata_db.get_chunks(book_id, chapter_id=cid)
                all_chunks.extend(chunks_dict["chunks"])
            return all_chunks

        if query_type == "chapter":
            return self.metadata_db.get_chunks(book_id, chapter_id=param)["chunks"]

        if query_type == "till_chapter":
            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            # chapters rows: (id, chapter_number, title, start_time, end_time, summary)
            target_ids = [c[0] for c in chapters if c[1] <= param]

            results = []
            for cid in target_ids:
                results.extend(self.metadata_db.get_chunks(book_id, cid)["chunks"])
            return results

        if query_type == "timestamp":
            all_chunks = self.metadata_db.get_chunks(book_id)["chunks"]
            # chunks rows: (id, chapter_id, start_time, end_time, text, token_count)
            return [c for c in all_chunks if c[2] <= param <= c[3]]

        return []

    # ------------------------------------------------------------
    # QUERY EXPANSION FOR BETTER RETRIEVAL
    # ------------------------------------------------------------
    def expand_query(self, query: str) -> List[str]:
        """"Generate query variations for better retrieval """

        # (1) REMOVE / COMMENT OUT the short-query early return
        # if len(query.split()) <= 3:
        #     logger.info("Simple Query, skipping expansion!")
        #     return [query]

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
            variations = [
                line.strip()
                for line in response.text.strip().split('\n')
                if line.strip()
            ]

            # Add original query
            all_queries = [query] + variations[:4]  # Original + 4 variations = 5 total
            return all_queries

        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [query]  # Fallback to original query

    # ------------------------------------------------------------
    # LLM ANSWER GENERATION
    # ------------------------------------------------------------
    def generate_answer(
        self,
        query: str,
        context_results: List[Dict],
        chat_history: List[Dict[str, str]] = None,
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

        # Detect if this is a "summary style" question
        is_chapter_summary = any(
            phrase in query_lower
            for phrase in [
                "what happens in chapter",
                "what happened in chapter",
                "summary of chapter",
                "summarize chapter",
                "summarise chapter",
                "can you summarize chapter",
                "what happens until chapter",
                "what happened until chapter",
                "summarize until chapter",
                "summarise until chapter",
                "until chapter",
                "till chapter",
            ]
        )

        if is_chapter_summary:
            # Narrative, structured summary style
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
            
            usage_metadata = {}
            if response.usage_metadata:
                usage_metadata = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }

            if response.candidates and response.candidates[0].content.parts:
                return response.text, usage_metadata
            else:
                logger.warning(f"Gemini returned no content. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'None'}")
                return "I apologize, but I couldn't generate an answer from the provided context.", usage_metadata
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error generating answer: {e}", {}

    # ------------------------------------------------------------
    # "WHEN" QUESTION RESOLVER
    # ------------------------------------------------------------
    def generate_when_answer(
        self,
        query: str,
        context_results: List[Dict],
        chat_history: List[Dict[str, str]] = None,
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
            if not resp.candidates or not resp.candidates[0].content.parts:
                return None
            
            raw = resp.text.strip()

            # Light cleanup for ```json fences
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            parsed = json.loads(raw)
            return parsed
        except Exception as e:
            logger.error(f"generate_when_answer error: {e}")
            return None

    # ------------------------------------------------------------
    # SPOILER-SAFE ANSWERS FOR TIMESTAMP QUERIES
    # ------------------------------------------------------------
    def generate_spoiler_safe_answer(
        self,
        query: str,
        passages: List[Dict],
        chat_history: List[Dict[str, str]] = None,
    ):
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
            if response.candidates and response.candidates[0].content.parts:
                return response.text.strip()
            return "No answer generated."
        except Exception as e:
            return f"LLM error: {e}"

    # ------------------------------------------------------------
    # MAIN QA HANDLER
    # ------------------------------------------------------------
    def ask_question(self, request: QueryRequest) -> QueryResponse:
        """
        Wrapper for MLflow tracking.
        """
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # End any dangling run
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=f"qa_{request.book_id or 'default'}"):
            mlflow.log_param("query", request.query)
            mlflow.log_param("book_id", request.book_id)
            mlflow.log_param("session_id", request.session_id)
            mlflow.log_param("is_timestamp_query", "when" in request.query.lower() or "time" in request.query.lower())
            
            start_time = time.time()
            try:
                response = self._internal_ask_question(request)
                
                duration = time.time() - start_time
                mlflow.log_metric("qa_duration_sec", duration)
                mlflow.log_text(response.answer, "answer.txt")
                mlflow.log_metric("citations_count", len(response.citations))
                if response.audio_references:
                    logger.info(f"Audio references: {response.audio_references}")
                    mlflow.log_metric("audio_refs_count", len(response.audio_references))

                if response.usage_metadata:
                    mlflow.log_metric("prompt_tokens", response.usage_metadata.get("prompt_tokens", 0))
                    mlflow.log_metric("completion_tokens", response.usage_metadata.get("completion_tokens", 0))
                    mlflow.log_metric("total_tokens", response.usage_metadata.get("total_tokens", 0))
                
                return response
            except Exception as e:
                mlflow.log_param("error", str(e))
                raise e

    def _internal_ask_question(self, request: QueryRequest) -> QueryResponse:
        book_id = request.book_id if request.book_id else "default"
        logger.info(f"QA for book_id={book_id}, query={request.query}")

        # "Progress" filters for spoiler-free up-to-chapter / up-to-time queries
        chapter_read = None
        completed_timestamp = None

        if getattr(request, "until_chapter", None):
            chapter_read = request.until_chapter

        if getattr(request, "until_time_seconds", None):
            completed_timestamp = request.until_time_seconds

        # -------------------------
        # Session Handling
        # -------------------------
        session_id = request.session_id
        chat_history: List[Dict[str, str]] = []
        if not session_id:
            session_id = self.metadata_db.create_session()
            logger.info(f"Created new session: {session_id}")
        else:
            chat_history = self.metadata_db.get_chat_history(session_id)
            logger.info(
                f"Retrieved {len(chat_history)} messages for session {session_id}"
            )

        start_time = time.time()

        query_type, param = self.parse_query(request.query)
        query_lower = request.query.lower()

        import re

        # Helper to pretty-print timestamps
        def fmt(sec: float) -> str:
            hh = int(sec // 3600)
            mm = int((sec % 3600) // 60)
            ss = int(sec % 60)
            return f"{hh:02d}:{mm:02d}:{ss:02d}"

        # ============================================================
        # "UNTIL CHAPTER N AT TIME"
        # e.g. "What happens until Chapter 4 at around 00:07:46?"
        # ============================================================
        m_until = re.search(
            r'(?:until|till)\s+chapter\s+(\d+).*?(?:at|around)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?',
            query_lower,
        )
        if m_until:
            target_chapter = int(m_until.group(1))
            # Parse timestamp (supports mm:ss or hh:mm:ss)
            if m_until.group(4):  # hh:mm:ss
                h = int(m_until.group(2))
                m = int(m_until.group(3))
                s = int(m_until.group(4))
            else:  # mm:ss
                h = 0
                m = int(m_until.group(2))
                s = int(m_until.group(3))
            target_ts = h * 3600 + m * 60 + s

            logger.info(
                f"[UNTIL] Up to Chapter {target_chapter} at {fmt(target_ts)} "
                f"for book_id={book_id}"
            )

            # 1) Get chapters, build mappings
            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            # chapters rows: (id_pk, chapter_number, title, start_time, end_time, summary)
            chap_pk_to_num = {row[0]: row[1] for row in chapters}
            chap_num_to_pk = {row[1]: row[0] for row in chapters}

            if target_chapter not in chap_num_to_pk:
                return QueryResponse(
                    answer=f"Chapter {target_chapter} not found.",
                    citations=[],
                    session_id=session_id,
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
                    session_id=session_id,
                )

            # 3) Convert chunks into context_results for LLM
            context_results = []
            for c in context_chunks:
                chap_pk = c[1]
                chap_num = chap_pk_to_num.get(chap_pk)
                context_results.append(
                    {
                        "metadata": {
                            "chapter_id": chap_num,
                            "start_time": c[2],
                            "end_time": c[3],
                            "text": c[4],
                            "token_count": c[5],
                        }
                    }
                )

            prefix = (
                f"**Summary up to Chapter {target_chapter} at {fmt(target_ts)}:**\n\n"
            )
            llm_answer, usage_metadata = self.generate_answer(
                request.query, context_results, chat_history
            )
            final_answer = prefix + llm_answer

            citations = [f"{c[2]}-{c[3]}" for c in context_chunks]

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(
                session_id, "assistant", final_answer
            )

            return QueryResponse(
                answer=final_answer,
                citations=citations,
                session_id=session_id,
                usage_metadata=usage_metadata,
            )

        # ============================================================
        # "UNTIL CHAPTER N" (NO TIMESTAMP)
        # e.g. "Can you summarize what happens until Chapter 3?"
        # ============================================================
        m_until_simple = re.search(
            r'(?:until|till)\s+chapter\s+(\d+)', query_lower
        )
        if m_until_simple:
            target_chapter = int(m_until_simple.group(1))
            logger.info(
                f"[UNTIL SIMPLE] Up to Chapter {target_chapter} for book_id={book_id}"
            )

            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            if not chapters:
                return QueryResponse(
                    answer="No chapter metadata found for this book.",
                    citations=[],
                    session_id=session_id,
                )

            # id_pk -> chapter_number
            chap_pk_to_num = {row[0]: row[1] for row in chapters}

            all_chunks = self.metadata_db.get_chunks(book_id)["chunks"]
            context_chunks = []
            for c in all_chunks:
                chap_pk = c[1]
                chap_num = chap_pk_to_num.get(chap_pk)
                if chap_num is None:
                    continue
                if chap_num <= target_chapter:
                    context_chunks.append(c)

            if not context_chunks:
                return QueryResponse(
                    answer=f"No content found up to Chapter {target_chapter}.",
                    citations=[],
                    session_id=session_id,
                )

            context_results = []
            for c in context_chunks:
                chap_num = chap_pk_to_num.get(c[1])
                context_results.append(
                    {
                        "metadata": {
                            "chapter_id": chap_num,
                            "start_time": c[2],
                            "end_time": c[3],
                            "text": c[4],
                            "token_count": c[5],
                        }
                    }
                )

            prefix = f"**Summary up to Chapter {target_chapter}:**\n\n"
            llm_answer, usage_metadata = self.generate_answer(
                request.query, context_results, chat_history
            )
            final_answer = prefix + llm_answer

            citations = [f"{c[2]}-{c[3]}" for c in context_chunks]

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(
                session_id, "assistant", final_answer
            )

            return QueryResponse(
                answer=final_answer,
                citations=citations,
                session_id=session_id,
                usage_metadata=usage_metadata,
            )

        # ============================================================
        # UNIVERSAL "CHAPTER + TIMESTAMP" QUERY
        # e.g. "What happens in Chapter 3 at 00:04:10?"
        # ============================================================
        m1 = re.search(
            r'chapter\s+(\d+).*?(\d{1,2}):(\d{2})(?::(\d{2}))?',
            query_lower,
        )
        m2 = re.search(
            r'(\d{1,2}):(\d{2})(?::(\d{2}))?.*?chapter\s+(\d+)',
            query_lower,
        )

        chapter_number = None
        timestamp_sec = None

        if m1:
            chapter_number = int(m1.group(1))
            if m1.group(4):  # hh:mm:ss
                h = int(m1.group(2))
                m = int(m1.group(3))
                s = int(m1.group(4))
            else:  # mm:ss
                h = 0
                m = int(m1.group(2))
                s = int(m1.group(3))
            timestamp_sec = h * 3600 + m * 60 + s

        elif m2:
            if m2.group(3):  # hh:mm:ss
                h = int(m2.group(1))
                m = int(m2.group(2))
                s = int(m2.group(3))
            else:  # mm:ss
                h = 0
                m = int(m2.group(1))
                s = int(m2.group(2))
            timestamp_sec = h * 3600 + m * 60 + s
            chapter_number = int(m2.group(4))

        if chapter_number is not None and timestamp_sec is not None:
            logger.info(
                f"[Timestamp+Chapter] chapter_number={chapter_number}, t={timestamp_sec}s"
            )

            chapters = self.metadata_db.get_chapters(book_id)["chapters"]
            # chapters: (id_pk, chapter_number, title, start_time, end_time, summary)
            num_to_pk = {c[1]: c[0] for c in chapters}

            if chapter_number not in num_to_pk:
                return QueryResponse(
                    answer=f"Chapter {chapter_number} not found.",
                    citations=[],
                    session_id=session_id,
                )

            chapter_db_id = num_to_pk[chapter_number]

            chapter_chunks = self.metadata_db.get_chunks(
                book_id, chapter_id=chapter_db_id
            )["chunks"]
            chapter_chunks = sorted(chapter_chunks, key=lambda x: x[2])

            match = [
                c for c in chapter_chunks
                if c[2] <= timestamp_sec <= c[3]
            ]

            if not match:
                return QueryResponse(
                    answer=(
                        f"No event found around {fmt(timestamp_sec)} "
                        f"in Chapter {chapter_number}."
                    ),
                    citations=[],
                    session_id=session_id,
                )

            chunk = match[0]
            start, end, text = chunk[2], chunk[3], chunk[4]

            prefix = (
                f"**In Chapter {chapter_number}, around {fmt(start)}–{fmt(end)}:**\n\n"
            )

            llm_answer, usage_metadata = self.generate_answer(
                request.query,
                [{"metadata": {"chapter_id": chapter_number,
                               "start_time": start,
                               "end_time": end,
                               "text": text,
                               "token_count": chunk[5]}}],
                chat_history,
            )

            final_answer = prefix + llm_answer

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(
                session_id, "assistant", final_answer
            )

            return QueryResponse(
                answer=final_answer,
                citations=[f"{start}-{end}"],
                session_id=session_id,
                audio_references=[{
                    "chapter_id": chapter_number,
                    "start_time": start,
                    "end_time": end
                }],
                usage_metadata=usage_metadata,
            )

        # ============================================================
        # SINGLE TIMESTAMP QUERY (NO CHAPTER)
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
                        "token_count": c[5],
                    },
                }
                for c in chunks
                if c[2] <= t <= c[3]
            ]

            if not matched:
                return QueryResponse(
                    answer=f"No content found around {fmt(t)}.",
                    citations=[],
                    session_id=session_id,
                )

            answer = self.generate_spoiler_safe_answer(
                request.query, matched, chat_history
            )
            citations = [
                f"{m['metadata']['start_time']}-{m['metadata']['end_time']}"
                for m in matched
            ]

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(
                session_id, "assistant", answer
            )
            
            # Construct audio references for all matched chunks
            audio_refs = []
            for m in matched:
                meta = m["metadata"]
                audio_refs.append({
                    "chapter_id": meta["chapter_id"],
                    "start_time": meta["start_time"],
                    "end_time": meta["end_time"]
                })

            return QueryResponse(
                answer=answer,
                citations=citations,
                session_id=session_id,
                audio_references=audio_refs
            )

        # ============================================================
        # TIMESTAMP RANGE QUERY
        # ============================================================
        if query_type == "timestamp_range":
            t1, t2 = param
            logger.info(f"[Timestamp Range] {t1}-{t2}")

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
                    },
                }
                for c in chunks
                if not (c[3] < t1 or c[2] > t2)
            ]

            if not matched:
                return QueryResponse(
                    answer="No content within that timestamp range.",
                    citations=[],
                    session_id=session_id,
                )

            answer = self.generate_spoiler_safe_answer(
                request.query, matched, chat_history
            )
            citations = [
                f"{m['metadata']['start_time']}-{m['metadata']['end_time']}"
                for m in matched
            ]

            # Save chat
            self.metadata_db.add_chat_message(session_id, "user", request.query)
            self.metadata_db.add_chat_message(
                session_id, "assistant", answer
            )

            return QueryResponse(
                answer=answer,
                citations=citations,
                session_id=session_id,
            )

        # ============================================================
        # VECTOR SEARCH (NORMAL FLOW) WITH CHAPTER/TIMESTAMP FILTERING
        # ============================================================
        vector_db = get_vector_db(book_id=book_id)

        # Skip expansion for timestamp/timestamp_range (already handled),
        # allow for general / chapter-only questions.
        if query_type in ["timestamp", "timestamp_range"]:
            expanded_queries = [request.query]
        else:
            expanded_queries = self.expand_query(request.query)

        logger.info(f"Expanded queries: {expanded_queries}")

        all_results = []
        for q in expanded_queries:
            embedding = embedding_model.encode(
                [q],
                normalize_embeddings=True,
            )[0].tolist()

            # IMPORTANT: this is where chapter_read + completed_timestamp are used
            # to avoid future spoilers and restrict context.
            results = vector_db.search(
                embedding,
                request.top_k,
                chapter_read,
                completed_timestamp,
            )
            all_results.extend(results)

        # Deduplicate by chunk text
        seen = set()
        unique_results = []
        for r in all_results:
            text = r["metadata"]["text"]
            if text not in seen:
                seen.add(text)
                unique_results.append(r)

        results = unique_results[: request.top_k]

        # Extra handling for explicit "chapters ..." queries:
        if query_type == "chapters" and param:
            target_chapters = param if isinstance(param, list) else [param]
            logger.info(f"Filtering results for chapters {target_chapters}")

            # SPOILER CHECK: If user has chapter_read limit, block requesting chapters beyond it
            if chapter_read is not None:
                forbidden_chapters = [c for c in target_chapters if c > chapter_read]
                if forbidden_chapters:
                    logger.warning(
                        f"User requested chapters {forbidden_chapters} but only read up to chapter {chapter_read}"
                    )
                    return QueryResponse(
                        answer=f"You've only read up to Chapter {chapter_read}. "
                               f"I can't show you information from Chapter {', '.join(map(str, forbidden_chapters))} "
                               f"to avoid spoilers. Please continue reading!",
                        citations=[],
                        session_id=session_id,
                    )

            # 1) Filter current vector search results
            filtered_vector = [
                r
                for r in results
                if r["metadata"].get("chapter_id") in target_chapters
            ]

            # 2) Identify missing chapters
            found_chapters = {
                r["metadata"].get("chapter_id") for r in filtered_vector
            }
            missing_chapters = [
                c for c in target_chapters if c not in found_chapters
            ]

            # 3) Fetch missing chapters directly from vector DB metadata
            if missing_chapters:
                logger.info(
                    f"Vector search missed chapters {missing_chapters}, "
                    f"fetching from Vector DB metadata"
                )
                for cid in missing_chapters:
                    chapter_metas = vector_db.get_by_chapter(cid)
                    for meta in chapter_metas:
                        filtered_vector.append({"score": 1.0, "metadata": meta})

            results = filtered_vector
            logger.info(
                f"Final results count for chapters {target_chapters}: {len(results)}"
            )

        if not results:
            return QueryResponse(
                answer="No relevant information found.",
                citations=[],
                session_id=session_id,
            )

        # Sort key: by chapter_id then start_time (for fallback)
        def chapter_key(r):
            cid = r["metadata"].get("chapter_id")
            if cid is None:
                cid = 10**9
            return (cid, r["metadata"].get("start_time", 0.0))

        # Detect natural-language "when" questions
        is_timestamp_question = any(
            kw in query_lower
            for kw in [
                "when",
                "what time",
                "at what time",
                "timestamp",
                "point in the book",
            ]
        )

        timestamp_prefix = ""
        results_for_citation = results

        # Decide context for LLM
        if is_timestamp_question:
            # Let LLM pick the best passage for the "when"
            when_info = self.generate_when_answer(
                request.query, results, chat_history
            )

            chosen = None
            if when_info:
                target_chapter = when_info.get("chapter_id")
                target_start = float(when_info.get("start_time", 0))
                target_end = float(when_info.get("end_time", 0))

                # Use overlap-based matching instead of exact comparison
                # This handles LLM rounding/approximation of timestamps
                best_match = None
                best_overlap = 0
                
                for r in results:
                    m = r["metadata"]
                    chunk_start = m.get("start_time", 0.0)
                    chunk_end = m.get("end_time", 0.0)
                    
                    # Check if same chapter and time ranges overlap
                    if m.get("chapter_id") == target_chapter:
                        # Calculate overlap duration
                        overlap_start = max(chunk_start, target_start)
                        overlap_end = min(chunk_end, target_end)
                        overlap_duration = max(0, overlap_end - overlap_start)
                        
                        # Track the chunk with maximum overlap
                        if overlap_duration > best_overlap:
                            best_overlap = overlap_duration
                            best_match = r
                
                if best_match is not None:
                    chosen = best_match
                    logger.info(
                        f"Matched chunk with {best_overlap:.1f}s overlap for timestamp question"
                    )

            # Fallback: earliest chunk
            if chosen is None:
                chosen = min(results, key=chapter_key)

            results_for_llm = [chosen]
            results_for_citation = [chosen]

            meta = chosen["metadata"]
            timestamp_prefix = (
                f"**Timestamp:** This occurs in **Chapter {meta.get('chapter_id')}**, "
                f"around **{fmt(meta.get('start_time', 0.0))}–{fmt(meta.get('end_time', 0.0))}**.\n\n"
            )
        else:
            # For general questions, use all retrieved chunks
            results_for_llm = results

        # Ask LLM for final answer
        llm_answer = self.generate_answer(
            request.query, results_for_llm, chat_history
        )

        # For non-timestamp questions, try to align citations with chapter mentioned by LLM
        if not is_timestamp_question:
            chapter_from_llm = None
            m_ch = re.search(r'[Cc]hapter\s+(\d+)', llm_answer)
            if m_ch:
                try:
                    chapter_from_llm = int(m_ch.group(1))
                    logger.info(
                        f"LLM explicitly mentioned Chapter {chapter_from_llm}"
                    )
                except ValueError:
                    chapter_from_llm = None

            if chapter_from_llm is not None:
                filtered = [
                    r
                    for r in results_for_citation
                    if r["metadata"].get("chapter_id") == chapter_from_llm
                ]
                if filtered:
                    logger.info(
                        f"Restricting citations to {len(filtered)} chunks "
                        f"from Chapter {chapter_from_llm}"
                    )
                    results_for_citation = filtered

            # Extra safeguard: apply explicit chapter filter from query
            if query_type == "chapters" and param:
                target = param if isinstance(param, list) else [param]
                results_for_citation = [
                    r
                    for r in results_for_citation
                    if r["metadata"].get("chapter_id") in target
                ]

            if not results_for_citation:
                return QueryResponse(
                    answer="No relevant information found.",
                    citations=[],
                    session_id=session_id,
                )

        # Final answer
        final_answer = timestamp_prefix + llm_answer

        citations = [
            f"{r['metadata'].get('start_time')}-{r['metadata'].get('end_time')}"
            for r in results_for_citation
        ]

        # Save chat history
        self.metadata_db.add_chat_message(session_id, "user", request.query)
        self.metadata_db.add_chat_message(
            session_id, "assistant", final_answer
        )

        total_time = time.time() - start_time
        logger.info(f"QA total time: {total_time:.3f}s")
        
        # Populate audio references for UI playback if it's a timestamp question
        audio_refs = []
        logger.info(f"DEBUG: is_timestamp_question={is_timestamp_question}, results_for_citation_len={len(results_for_citation)}")
        if is_timestamp_question and results_for_citation:
             for r in results_for_citation:
                 m = r["metadata"]
                 audio_refs.append({
                     "chapter_id": m.get("chapter_id"),
                     "start_time": m.get("start_time"),
                     "end_time": m.get("end_time")
                 })
        logger.info(f"DEBUG: Generated audio_refs: {audio_refs}")

        return QueryResponse(
            answer=final_answer,
            citations=citations,
            session_id=session_id,
            audio_references=audio_refs
        )

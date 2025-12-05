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

logger = logging.getLogger(__name__)

from services.storage.metadata_db_service import MetadataDBService

# Load embedding model
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

from core.model_loader import get_embedding_model

embedding_model = get_embedding_model()

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

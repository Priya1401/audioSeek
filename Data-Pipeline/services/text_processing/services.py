import glob
import json
import logging
import os
import sqlite3
from typing import Dict, Any, List

from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from config import settings
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
from vector_db_interface import VectorDBInterface

logger = logging.getLogger(__name__)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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


class ChunkingService:
    """Service for chunking transcripts"""

    @staticmethod
    def _get_file_list(request: ChunkingRequest) -> List[str]:
        files = []

        if request.file_path:
            if not os.path.exists(request.file_path):
                raise HTTPException(404, f"File not found: {request.file_path}")
            files = [request.file_path]

        elif request.file_paths:
            for fp in request.file_paths:
                if not os.path.exists(fp):
                    raise HTTPException(404, f"File not found: {fp}")
            files = request.file_paths

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

        if request.output_file:
            with open(request.output_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2)
            response["output_file"] = request.output_file

        return ChunkResponse(**response)


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

            embeddings = embedding_model.encode(texts).tolist()

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


class MetadataDBService:
    """Book-aware metadata database operations"""

    def __init__(self, db_path="audiobook_metadata.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            with open("schema.sql", "r") as f:
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

    def generate_answer(self, query: str, context_texts: List[str]):
        context = "\n".join(
            [f"Passage {i + 1}:\n{text}\n" for i, text in
             enumerate(context_texts)]
        )
        prompt = f"Answer based on the context only:\n\n{context}\n\nQuestion: {query}\nAnswer:"

        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error generating answer: {e}"

    def ask_question(self, request: QueryRequest):
        book_id = request.book_id if request.book_id else "default"
        logger.info(f"QA for book_id: {book_id}, query: {request.query}")

        query_type, param = self.parse_query(request.query)

        if query_type in ["chapter", "till_chapter", "timestamp"]:
            chunks = self.get_chunks_from_metadata(query_type, param, book_id)

            if not chunks:
                return QueryResponse(answer="No relevant information found.",
                                     citations=[])

            texts = [c[4] for c in chunks]
            citations = [f"{c[2]}-{c[3]}" for c in chunks]

        else:
            vector_db = get_vector_db(book_id=book_id)
            embedding = embedding_model.encode([request.query])[0].tolist()

            results = vector_db.search(embedding, request.top_k)

            if not results:
                return QueryResponse(answer="No relevant information found.",
                                     citations=[])

            texts = [r["metadata"]["text"] for r in results]
            citations = [
                f"{r['metadata'].get('start_time')}-{r['metadata'].get('end_time')}"
                for r in results
            ]

        answer = self.generate_answer(request.query, texts)
        return QueryResponse(answer=answer, citations=citations)


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

    # @staticmethod
    # def process_full_pipeline(
    #     request: FullPipelineRequest) -> FullPipelineResponse:
    #     book_id = extract_book_id_from_path(
    #         book_id=request.book_id,
    #         folder_path=request.folder_path,
    #         file_path=request.file_path,
    #         file_paths=request.file_paths
    #     )
    #     logger.info(f"Running full pipeline for book_id={book_id}")
    #
    #     metadata_db = MetadataDBService()
    #
    #     metadata_db.create_audiobook(
    #         book_id=book_id,
    #         title=book_id
    #     )
    #
    #     chunk_request = ChunkingRequest(
    #         book_id=book_id,
    #         file_path=request.file_path,
    #         file_paths=request.file_paths,
    #         folder_path=request.folder_path,
    #         target_tokens=request.target_tokens,
    #         overlap_tokens=request.overlap_tokens
    #     )
    #     chunk_response = ChunkingService.chunk_transcript(chunk_request)
    #
    #     chapter_ids = {}
    #     for chapter in chunk_response.chapters:
    #         chapter_number = chapter.get("id", 0)
    #
    #         cid = metadata_db.create_chapter(
    #             book_id=book_id,
    #             chapter_number=chapter_number,
    #             title=chapter.get("title", f"Chapter {chapter_number}"),
    #             start_time=chapter.get("start_time", 0.0),
    #             end_time=chapter.get("end_time", 0.0),
    #             summary=chapter.get("summary")
    #         )
    #
    #         chapter_ids[chapter.get("id")] = cid
    #
    #     for chunk in chunk_response.chunks:
    #         metadata_db.create_chunk(
    #             book_id=book_id,
    #             chapter_id=chapter_ids.get(chunk.get("chapter_id")),
    #             text=chunk["text"],
    #             start_time=chunk["start_time"],
    #             end_time=chunk["end_time"],
    #             token_count=chunk["token_count"],
    #             source_file=chunk.get("source_file")
    #         )
    #
    #     texts = [c["text"] for c in chunk_response.chunks]
    #     embeddings = embedding_model.encode(texts).tolist()
    #
    #     vector_db_added = False
    #     if request.add_to_vector_db:
    #         vector_db = get_vector_db(book_id=book_id)
    #
    #         metadatas = [
    #             {
    #                 "text": c["text"],
    #                 "start_time": c["start_time"],
    #                 "end_time": c["end_time"],
    #                 "chapter_id": c.get("chapter_id"),
    #                 "token_count": c["token_count"],
    #                 "source_file": c.get("source_file"),
    #             }
    #             for c in chunk_response.chunks
    #         ]
    #
    #         vector_db.add_documents(embeddings, metadatas)
    #         vector_db_added = True
    #
    #     return FullPipelineResponse(
    #         book_id=book_id,
    #         chunks_count=len(chunk_response.chunks),
    #         chapters_count=len(chunk_response.chapters),
    #         entities_count=len(chunk_response.entities),
    #         embeddings_count=len(embeddings),
    #         vector_db_added=vector_db_added,
    #         files_processed=len(chunk_response.processed_files),
    #         message=f"Full pipeline completed for book_id={book_id}"
    #     )

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

        metadata_db = MetadataDBService()
        metadata_db.create_audiobook(book_id=book_id, title=book_id)

        chunk_request = ChunkingRequest(
            book_id=book_id,
            file_path=request.file_path,
            file_paths=request.file_paths,
            folder_path=request.folder_path,
            target_tokens=request.target_tokens,
            overlap_tokens=request.overlap_tokens
        )
        chunk_response = ChunkingService.chunk_transcript(chunk_request)

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

        texts = [c["text"] for c in chunk_response.chunks]
        embeddings = embedding_model.encode(texts).tolist()

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
            vector_db_added = True

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
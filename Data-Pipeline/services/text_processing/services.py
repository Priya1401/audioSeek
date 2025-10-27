import logging
import json
import os
import glob
import numpy as np
import faiss
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

from utils import (
    parse_transcript,
    detect_chapters,
    chunk_text,
    collect_unique_entities
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
    AudiobookCreate,
    ChapterCreate,
    ChunkCreate,
    EntityCreate,
    EntityMentionCreate,
    QueryResponse
)

logger = logging.getLogger(__name__)

# Load embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index (dimension 384 for all-MiniLM-L6-v2)
vector_index = faiss.IndexFlatIP(384)
vector_metadata = []

class ChunkingService:
    """Service for chunking transcripts"""
    
    @staticmethod
    def _get_file_list(request: ChunkingRequest) -> List[str]:
        """Get list of files to process based on request"""
        files = []
        
        # Single file
        if request.file_path:
            if not os.path.exists(request.file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
            files = [request.file_path]
        
        # Multiple files
        elif request.file_paths:
            for fp in request.file_paths:
                if not os.path.exists(fp):
                    raise HTTPException(status_code=404, detail=f"File not found: {fp}")
            files = request.file_paths
        
        # Folder
        elif request.folder_path:
            if not os.path.exists(request.folder_path):
                raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")
            
            # Get all .txt files in folder
            pattern = os.path.join(request.folder_path, "*.txt")
            files = glob.glob(pattern)
            
            if not files:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No .txt files found in folder: {request.folder_path}"
                )
        
        # Validate all files are .txt
        for f in files:
            if not f.endswith('.txt'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only .txt files are allowed, found: {f}"
                )
        
        return files
    
    @staticmethod
    def _process_single_file(file_path: str, target_tokens: int, overlap_tokens: int) -> Dict[str, Any]:
        """Process a single transcript file"""
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file {file_path}: {str(e)}")
        
        segments = parse_transcript(transcript)
        
        if not segments:
            logger.warning(f"No valid segments found in {file_path}")
            return {
                'chunks': [],
                'chapters': [],
                'entities': [],
                'file': file_path
            }
        
        logger.info(f"Processing {len(segments)} segments from {file_path}")
        chapters = detect_chapters(segments)
        chunks = chunk_text(segments, target_tokens, overlap_tokens, chapters)
        
        # Add source file to each chunk
        for chunk in chunks:
            chunk['source_file'] = os.path.basename(file_path)
        
        logger.info(f"Generated {len(chunks)} chunks from {file_path}")
        
        all_entities = collect_unique_entities(chunks)
        
        return {
            'chunks': chunks,
            'chapters': chapters,
            'entities': all_entities,
            'file': file_path
        }
    
    @staticmethod
    def chunk_transcript(request: ChunkingRequest) -> ChunkResponse:
        """Process transcript file(s) and generate chunks"""
        files = ChunkingService._get_file_list(request)
        
        logger.info(f"Processing {len(files)} file(s)")
        
        # Aggregate results from all files
        all_chunks = []
        all_chapters = []
        all_entities = {}
        processed_files = []
        
        for file_path in files:
            try:
                result = ChunkingService._process_single_file(
                    file_path, 
                    request.target_tokens, 
                    request.overlap_tokens
                )
                
                all_chunks.extend(result['chunks'])
                all_chapters.extend(result['chapters'])
                
                # Merge entities (avoid duplicates)
                for entity in result['entities']:
                    entity_key = (entity['name'], entity['type'])
                    if entity_key not in all_entities:
                        all_entities[entity_key] = entity
                
                processed_files.append(file_path)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                # Continue processing other files
                continue
        
        if not all_chunks:
            raise HTTPException(
                status_code=400, 
                detail="No valid chunks generated from any of the files"
            )
        
        # Convert dict back to list for JSON serialization
        all_entities_list = list(all_entities.values())
        
        logger.info(f"Total: {len(all_chunks)} chunks from {len(processed_files)} files")
        
        response_data = {
            'chunks': all_chunks,
            'chapters': all_chapters,
            'entities': all_entities_list,
            'processed_files': processed_files
        }
        
        if request.output_file:
            try:
                with open(request.output_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2)
                response_data['output_file'] = request.output_file
                logger.info(f"Results saved to {request.output_file}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving output file: {str(e)}")
        
        return ChunkResponse(**response_data)


class EmbeddingService:
    """Service for generating embeddings"""
    
    @staticmethod
    def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for text chunks"""
        try:
            texts = []
            
            if request.chunks_file:
                if not os.path.exists(request.chunks_file):
                    raise HTTPException(status_code=404, detail=f"File not found: {request.chunks_file}")
                
                logger.info(f"Reading chunks from file: {request.chunks_file}")
                with open(request.chunks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                texts = [chunk['text'] for chunk in data.get('chunks', [])]
                logger.info(f"Extracted {len(texts)} texts from chunks file")
            
            elif request.texts:
                texts = request.texts
            
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Either 'texts' or 'chunks_file' must be provided"
                )
            
            if not texts:
                raise HTTPException(status_code=400, detail="No texts found to embed")
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = embedding_model.encode(texts).tolist()
            logger.info("Embeddings generated successfully")
            
            response_data = {
                'embeddings': embeddings,
                'count': len(embeddings)
            }
            
            if request.output_file:
                try:
                    with open(request.output_file, 'w', encoding='utf-8') as f:
                        json.dump(response_data, f, indent=2)
                    response_data['output_file'] = request.output_file
                    logger.info(f"Embeddings saved to {request.output_file}")
                except Exception as e:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Error saving output file: {str(e)}"
                    )
            
            return EmbeddingResponse(**response_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


class VectorDBService:
    """Service for vector database operations"""

    @staticmethod
    def add_from_files(request) -> AddFromFilesResponse:
        """Read chunks and embeddings from files and add to vector DB"""
        import json
        
        # Read chunks file
        with open(request.chunks_file, 'r') as f:
            chunks_data = json.load(f)
        chunks = chunks_data['chunks']
        
        # Read embeddings file
        with open(request.embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        embeddings = embeddings_data['embeddings']
        
        if len(chunks) != len(embeddings):
            raise HTTPException(
                status_code=400,
                detail=f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        
        # Prepare metadata
        metadatas = [
            {
                'text': chunk['text'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'chapter_id': chunk.get('chapter_id'),
                'token_count': chunk['token_count'],
                'source_file': chunk.get('source_file')
            }
            for chunk in chunks
        ]
        
        # Add to vector DB
        add_request = AddDocumentsRequest(
            embeddings=embeddings,
            metadatas=metadatas
        )
        VectorDBService.add_documents(add_request)
        
        return AddFromFilesResponse(
            message=f"Added {len(chunks)} documents from files to vector DB",
            chunks_count=len(chunks),
            embeddings_count=len(embeddings)
        )
    
    @staticmethod
    def add_documents(request: AddDocumentsRequest) -> AddDocumentsResponse:
        """Add documents to the vector database"""
        if len(request.embeddings) != len(request.metadatas):
            raise HTTPException(
                status_code=400, 
                detail="Embeddings and metadatas length mismatch"
            )
        
        logger.info(f"Adding {len(request.embeddings)} documents to vector DB")
        vectors = np.array(request.embeddings, dtype=np.float32)
        vector_index.add(vectors)
        vector_metadata.extend(request.metadatas)
        logger.info("Documents added successfully")
        
        return AddDocumentsResponse(
            message=f"Added {len(request.embeddings)} documents",
            count=len(request.embeddings)
        )
    
    @staticmethod
    def search(request: SearchRequest) -> SearchResponse:
        """Search for similar vectors in the database"""
        logger.info(f"Searching for top {request.top_k} results")
        query = np.array([request.query_embedding], dtype=np.float32)
        distances, indices = vector_index.search(query, request.top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(vector_metadata):
                results.append({
                    "metadata": vector_metadata[idx],
                    "score": float(distances[0][i])
                })
        
        logger.info(f"Found {len(results)} results")
        return SearchResponse(results=results, count=len(results))
    
    @staticmethod
    def query_text(request: QueryRequest) -> SearchResponse:
        """Query the vector database with text (generates embedding automatically)"""
        logger.info(f"Generating embedding for query text")
        query_embedding = embedding_model.encode([request.query])[0].tolist()
        
        search_request = SearchRequest(
            query_embedding=query_embedding,
            top_k=request.top_k
        )
        
        return VectorDBService.search(search_request)
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get vector database statistics"""
        return {
            "service": "Vector DB Service",
            "status": "healthy",
            "documents_count": len(vector_metadata),
            "index_total": vector_index.ntotal
        }


class PipelineService:
    """Service for running the full pipeline"""
    
    @staticmethod
    def process_combined_pipeline(request: CombinedRequest) -> CombinedResponse:
        """Run chunking and embedding pipeline"""
        logger.info("Starting combined pipeline processing")
        
        chunk_request = ChunkingRequest(
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
                with open(request.embeddings_output_file, 'w', encoding='utf-8') as f:
                    json.dump(embedding_data, f, indent=2)
                response_data['embeddings_output_file'] = request.embeddings_output_file
                logger.info(f"Embeddings saved to {request.embeddings_output_file}")
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error saving embeddings file: {str(e)}"
                )
        
        return CombinedResponse(**response_data)
    
    @staticmethod
    def process_full_pipeline(request: FullPipelineRequest) -> FullPipelineResponse:
        """Run the complete pipeline: chunk, embed, and add to vector DB"""
        logger.info("Starting full pipeline processing (chunk + embed + vector DB)")
        
        chunk_request = ChunkingRequest(
            file_path=request.file_path,
            file_paths=request.file_paths,
            folder_path=request.folder_path,
            target_tokens=request.target_tokens,
            overlap_tokens=request.overlap_tokens
        )
        chunk_response = ChunkingService.chunk_transcript(chunk_request)
        
        texts = [chunk['text'] for chunk in chunk_response.chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = embedding_model.encode(texts).tolist()
        
        vector_db_added = False
        if request.add_to_vector_db:
            metadatas = [
                {
                    'text': chunk['text'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'chapter_id': chunk.get('chapter_id'),
                    'token_count': chunk['token_count'],
                    'source_file': chunk.get('source_file')
                }
                for chunk in chunk_response.chunks
            ]
            
            add_request = AddDocumentsRequest(
                embeddings=embeddings,
                metadatas=metadatas
            )
            VectorDBService.add_documents(add_request)
            vector_db_added = True
            logger.info("Documents added to vector DB")
        
        return FullPipelineResponse(
            chunks_count=len(chunk_response.chunks),
            chapters_count=len(chunk_response.chapters),
            entities_count=len(chunk_response.entities),
            embeddings_count=len(embeddings),
            vector_db_added=vector_db_added,
            files_processed=len(chunk_response.processed_files),
            message="Full pipeline completed successfully"
        )


class MetadataDBService:
    """Service for metadata database operations"""

    def __init__(self, db_path="audiobook_metadata.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if tables already exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audiobooks'")
        if cursor.fetchone():
            logger.info("Database already exists, skipping schema creation")
            conn.close()
            return
        
        # Only create schema if tables don't exist
        try:
            with open("schema.sql", "r") as f:
                schema = f.read()
            cursor.executescript(schema)
            conn.commit()
            logger.info("Database schema created successfully")
        except FileNotFoundError:
            logger.warning("schema.sql not found, creating basic schema")
            # Fallback basic schema if file not found
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS audiobooks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    author TEXT,
                    duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS chapters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audiobook_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    summary TEXT,
                    FOREIGN KEY (audiobook_id) REFERENCES audiobooks(id)
                );
                
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audiobook_id INTEGER NOT NULL,
                    chapter_id INTEGER,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER,
                    embedding_id INTEGER,
                    source_file TEXT,
                    FOREIGN KEY (audiobook_id) REFERENCES audiobooks(id),
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
                );
                
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    audiobook_id INTEGER NOT NULL,
                    FOREIGN KEY (audiobook_id) REFERENCES audiobooks(id)
                );
                
                CREATE TABLE IF NOT EXISTS entity_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id INTEGER NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    start_pos INTEGER,
                    end_pos INTEGER,
                    FOREIGN KEY (entity_id) REFERENCES entities(id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                );
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Error creating database schema: {str(e)}")
            conn.rollback()
        finally:
            conn.close()


class QAService:
    """Service for question answering"""

    def __init__(self, metadata_db: MetadataDBService, vector_db: VectorDBService, embedding_model):
        self.metadata_db = metadata_db
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        import os
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def parse_query(self, query: str):
        import re
        query_lower = query.lower()
        if "till" in query_lower and "chapter" in query_lower:
            match = re.search(r'till chapter (\d+)', query_lower)
            if match:
                return "till_chapter", int(match.group(1))
        elif "chapter" in query_lower:
            match = re.search(r'chapter (\d+)', query_lower)
            if match:
                return "chapter", int(match.group(1))
        elif "at" in query_lower:
            match = re.search(r'at (\d+):(\d+):(\d+)', query_lower)
            if match:
                h, m, s = map(int, match.groups())
                time_sec = h * 3600 + m * 60 + s
                return "timestamp", time_sec
        return "general", None

    def get_chunks_from_metadata(self, query_type, param, audiobook_id=1):
        if query_type == "chapter":
            result = self.metadata_db.get_chunks(audiobook_id, param)
            chunks = result["chunks"]
        elif query_type == "till_chapter":
            chapters_result = self.metadata_db.get_chapters(audiobook_id)
            chapters = chapters_result["chapters"]
            relevant_chapter_ids = [c[0] for c in chapters if c[1] <= param]
            chunks = []
            for cid in relevant_chapter_ids:
                result = self.metadata_db.get_chunks(audiobook_id, cid)
                chunks.extend(result["chunks"])
        elif query_type == "timestamp":
            result = self.metadata_db.get_chunks(audiobook_id)
            all_chunks = result["chunks"]
            chunks = [c for c in all_chunks if c[3] <= param <= c[4]]
        else:
            chunks = []
        return chunks

    def generate_answer(self, query: str, context_texts):
        context = "\n".join([f"Text {i+1}: {text}" for i, text in enumerate(context_texts)])
        prompt = f"Answer the question based on the following context from the audiobook:\n{context}\n\nQuestion: {query}\nAnswer:"

        import openai
        client = openai.OpenAI(api_key=self.openai_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def ask_question(self, request: QueryRequest, audiobook_id=1):
        query_type, param = self.parse_query(request.query)

        if query_type in ["chapter", "till_chapter", "timestamp"]:
            metadata_chunks = self.get_chunks_from_metadata(query_type, param, audiobook_id)
            if not metadata_chunks:
                return QueryResponse(answer="No relevant information found for the specified chapter/timestamp.", citations=[])
            context_texts = [chunk[5] for chunk in metadata_chunks]
            citations = [f"{chunk[3]:.2f}-{chunk[4]:.2f}" for chunk in metadata_chunks]
        else:
            query_embedding = self.embedding_model.encode([request.query])[0].tolist()
            search_request = SearchRequest(query_embedding=query_embedding, top_k=request.top_k)
            search_result = self.vector_db.search(search_request)
            if not search_result.results:
                return QueryResponse(answer="No relevant information found.", citations=[])
            results = search_result.results
            results.sort(key=lambda x: x["score"], reverse=True)
            chunks = results[:request.top_k]
            context_texts = [chunk['metadata']['text'] for chunk in chunks]
            citations = [f"{chunk['metadata'].get('formatted_start_time', 'Unknown')}-{chunk['metadata'].get('formatted_end_time', 'Unknown')}" for chunk in chunks]

        answer = self.generate_answer(request.query, context_texts)
        return QueryResponse(answer=answer, citations=citations)
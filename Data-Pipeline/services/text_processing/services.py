import logging
import json
import os
import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

from utils import (
    parse_transcript, 
    detect_chapters, 
    chunk_text, 
    collect_unique_entities
)
from models import (
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
    FullPipelineResponse
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
    def chunk_transcript(request: ChunkingRequest) -> ChunkResponse:
        """Process transcript file and generate chunks"""
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        if not request.file_path.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are allowed")
        
        try:
            with open(request.file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        segments = parse_transcript(transcript)
        
        if not segments:
            raise HTTPException(status_code=400, detail="No valid transcript segments found in file")
        
        logger.info(f"Processing {len(segments)} segments")
        chapters = detect_chapters(segments)
        chunks = chunk_text(segments, request.target_tokens, request.overlap_tokens, chapters)
        logger.info(f"Generated {len(chunks)} chunks")
        
        all_entities = collect_unique_entities(chunks)
        
        response_data = {
            'chunks': chunks,
            'chapters': chapters,
            'entities': all_entities
        }
        
        if request.output_file:
            try:
                with open(request.output_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2)
                response_data['output_file'] = request.output_file
                logger.info(f"Chunks saved to {request.output_file}")
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
        query_embedding = embedding_model.encode([request.query_text])[0].tolist()
        
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
                    'token_count': chunk['token_count']
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
            message="Full pipeline completed successfully"
        )
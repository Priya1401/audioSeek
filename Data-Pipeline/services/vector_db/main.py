import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vector DB Service", description="Manage vector database with FAISS")

# In-memory FAISS index (dimension 384 for all-MiniLM-L6-v2)
index = faiss.IndexFlatIP(384)
metadata = []  # list of metadata dicts

class AddDocumentsRequest(BaseModel):
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]

class SearchRequest(BaseModel):
    query_embedding: List[float]
    top_k: int = 5

@app.post("/add")
async def add_documents(request: AddDocumentsRequest):
    if len(request.embeddings) != len(request.metadatas):
        raise HTTPException(status_code=400, detail="Embeddings and metadatas length mismatch")
    
    logger.info(f"Adding {len(request.embeddings)} documents to vector DB")
    vectors = np.array(request.embeddings, dtype=np.float32)
    index.add(vectors)
    metadata.extend(request.metadatas)
    logger.info("Documents added successfully")
    return {"message": f"Added {len(request.embeddings)} documents"}

@app.post("/search")
async def search(request: SearchRequest):
    logger.info(f"Searching for top {request.top_k} results")
    query = np.array([request.query_embedding], dtype=np.float32)
    distances, indices = index.search(query, request.top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "metadata": metadata[idx],
                "score": float(distances[0][i])
            })
    logger.info(f"Found {len(results)} results")
    return {"results": results}

@app.get("/")
async def root():
    return {"service": "Vector DB Service", "status": "healthy", "documents": len(metadata)}
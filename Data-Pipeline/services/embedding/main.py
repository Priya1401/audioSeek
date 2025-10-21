import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service", description="Generate embeddings for text chunks")

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingRequest(BaseModel):
    texts: List[str] = None
    chunks_file: str = None  # Path to chunking output JSON file
    output_file: str = None  # Optional: save embeddings to file

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    count: int
    output_file: str = None

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    try:
        texts = []
        
        # If chunks_file is provided, read it
        if request.chunks_file:
            if not os.path.exists(request.chunks_file):
                raise HTTPException(status_code=404, detail=f"File not found: {request.chunks_file}")
            
            logger.info(f"Reading chunks from file: {request.chunks_file}")
            with open(request.chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract text from chunks
            texts = [chunk['text'] for chunk in data.get('chunks', [])]
            logger.info(f"Extracted {len(texts)} texts from chunks file")
        
        # If texts are provided directly, use them
        elif request.texts:
            texts = request.texts
        
        else:
            raise HTTPException(status_code=400, detail="Either 'texts' or 'chunks_file' must be provided")
        
        if not texts:
            raise HTTPException(status_code=400, detail="No texts found to embed")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = model.encode(texts).tolist()
        logger.info("Embeddings generated successfully")
        
        response_data = {
            'embeddings': embeddings,
            'count': len(embeddings)
        }
        
        # Save to file if output_file is provided
        if request.output_file:
            try:
                with open(request.output_file, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2)
                response_data['output_file'] = request.output_file
                logger.info(f"Embeddings saved to {request.output_file}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving output file: {str(e)}")
        
        return EmbeddingResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"service": "Embedding Service", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
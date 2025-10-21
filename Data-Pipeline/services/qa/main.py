import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QA Service", description="Question answering with RAG")

# Service URLs (assume running locally)
EMBEDDING_URL = "http://localhost:8001/embed"
VECTOR_DB_URL = "http://localhost:8002/search"
METADATA_DB_URL = "http://localhost:8006"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUDIOBOOK_ID = 1  # Assume single audiobook for now

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

def get_embedding(text: str):
    response = requests.post(EMBEDDING_URL, json={"texts": [text]})
    response.raise_for_status()
    return response.json()["embeddings"][0]

def search_vector_db(embedding, top_k):
    response = requests.post(VECTOR_DB_URL, json={"query_embedding": embedding, "top_k": top_k})
    response.raise_for_status()
    return response.json()["results"]

def parse_query(query: str):
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

def get_chunks_from_metadata(query_type, param):
    if query_type == "chapter":
        response = requests.get(f"{METADATA_DB_URL}/chunks?audiobook_id={AUDIOBOOK_ID}&chapter_id={param}")
        chunks = response.json()["chunks"]
    elif query_type == "till_chapter":
        # Get all chapters up to param
        chapters_response = requests.get(f"{METADATA_DB_URL}/audiobooks/{AUDIOBOOK_ID}/chapters")
        chapters = chapters_response.json()["chapters"]
        relevant_chapter_ids = [c[0] for c in chapters if c[1] <= param]  # id, title, etc.
        chunks = []
        for cid in relevant_chapter_ids:
            response = requests.get(f"{METADATA_DB_URL}/chunks?audiobook_id={AUDIOBOOK_ID}&chapter_id={cid}")
            chunks.extend(response.json()["chunks"])
    elif query_type == "timestamp":
        # Find chunk at time
        response = requests.get(f"{METADATA_DB_URL}/chunks?audiobook_id={AUDIOBOOK_ID}")
        all_chunks = response.json()["chunks"]
        chunks = [c for c in all_chunks if c[3] <= param <= c[4]]  # start_time <= param <= end_time
    else:
        chunks = []
    return chunks

def generate_answer(query: str, context_texts):
    context = "\n".join([f"Text {i+1}: {text}" for i, text in enumerate(context_texts)])
    prompt = f"Answer the question based on the following context from the audiobook:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        query_type, param = parse_query(request.query)
        
        if query_type in ["chapter", "till_chapter", "timestamp"]:
            # Use metadata to get specific chunks
            metadata_chunks = get_chunks_from_metadata(query_type, param)
            if not metadata_chunks:
                return {"answer": "No relevant information found for the specified chapter/timestamp."}
            context_texts = [chunk[5] for chunk in metadata_chunks]  # text column
            citations = [f"{chunk[3]:.2f}-{chunk[4]:.2f}" for chunk in metadata_chunks]  # start-end times
        else:
            # General query: use vector search
            embedding = get_embedding(request.query)
            results = search_vector_db(embedding, request.top_k)
            if not results:
                return {"answer": "No relevant information found."}
            results.sort(key=lambda x: x["score"], reverse=True)
            chunks = results[:request.top_k]
            context_texts = [chunk['metadata']['text'] for chunk in chunks]
            citations = [f"{chunk['metadata'].get('formatted_start_time', 'Unknown')}-{chunk['metadata'].get('formatted_end_time', 'Unknown')}" for chunk in chunks]
        
        answer = generate_answer(request.query, context_texts)
        logger.info("Answer generated")
        
        return {"answer": answer, "citations": citations}
    except Exception as e:
        logger.error(f"Error in QA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"service": "QA Service", "status": "healthy"}
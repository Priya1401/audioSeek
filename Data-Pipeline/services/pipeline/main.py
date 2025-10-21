from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
import json

app = FastAPI(title="Pipeline Service", description="Orchestrate full audiobook processing")

# Service URLs
TRANSCRIPTION_URL = "http://localhost:8000/transcribe"
CHUNKING_URL = "http://localhost:8003/chunk"
EMBEDDING_URL = "http://localhost:8001/embed"
VECTOR_DB_URL = "http://localhost:8002/add"
METADATA_DB_URL = "http://localhost:8006"

@app.post("/process_audiobook")
async def process_audiobook(file: UploadFile = File(...), title: str = "Unknown Audiobook", author: str = None):
    try:
        # 1. Transcribe
        files = {"file": (file.filename, await file.read(), file.content_type)}
        trans_response = requests.post(TRANSCRIPTION_URL, files=files)
        trans_response.raise_for_status()
        trans_data = trans_response.json()
        transcript = trans_data["transcript"]
        duration = trans_data.get("duration")
        chapters_basic = trans_data.get("chapters", [])
        
        # 2. Create audiobook in metadata DB
        audiobook_resp = requests.post(METADATA_DB_URL + "/audiobooks", json={"title": title, "author": author, "duration": duration})
        audiobook_resp.raise_for_status()
        audiobook_id = audiobook_resp.json()["audiobook_id"]
        
        # 3. Chunk
        chunk_response = requests.post(CHUNKING_URL, json={"transcript": transcript})
        chunk_response.raise_for_status()
        chunk_data = chunk_response.json()
        chunks = chunk_data["chunks"]
        chapters = chunk_data["chapters"]
        entities = chunk_data["entities"]
        
        # 4. Store chapters in metadata DB
        for chap in chapters:
            chap_resp = requests.post(METADATA_DB_URL + "/chapters", json={
                "audiobook_id": audiobook_id,
                "title": chap["title"],
                "start_time": chap["start_time"],
                "end_time": chap["end_time"]
            })
            chap_resp.raise_for_status()
            chap["id"] = chap_resp.json()["chapter_id"]
        
        # 5. Store entities
        for ent in entities:
            ent_resp = requests.post(METADATA_DB_URL + "/entities", json={
                "name": ent["name"],
                "type": ent["type"],
                "audiobook_id": audiobook_id
            })
            ent_resp.raise_for_status()
            ent["id"] = ent_resp.json()["entity_id"]
        
        # 6. Extract texts and embed
        texts = [chunk["text"] for chunk in chunks]
        embed_response = requests.post(EMBEDDING_URL, json={"texts": texts})
        embed_response.raise_for_status()
        embeddings = embed_response.json()["embeddings"]
        
        # 7. Store chunks in metadata DB and prepare for vector DB
        metadatas = []
        for i, chunk in enumerate(chunks):
            # Find chapter_id
            chapter_id = None
            for chap in chapters:
                if chap["start_time"] <= chunk["start_time"] < chap["end_time"]:
                    chapter_id = chap["id"]
                    break
            
            chunk_resp = requests.post(METADATA_DB_URL + "/chunks", json={
                "audiobook_id": audiobook_id,
                "chapter_id": chapter_id,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "embedding_id": i  # index in FAISS
            })
            chunk_resp.raise_for_status()
            chunk_id = chunk_resp.json()["chunk_id"]
            
            # Store entity mentions (simplified, assume positions)
            for ent in chunk["entities"]:
                ent_id = next((e["id"] for e in entities if e["name"] == ent["name"]), None)
                if ent_id:
                    requests.post(METADATA_DB_URL + "/entity_mentions", json={
                        "entity_id": ent_id,
                        "chunk_id": chunk_id,
                        "start_pos": 0,  # placeholder
                        "end_pos": len(ent["name"])
                    })
            
            metadata = {
                "chunk_id": chunk_id,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "formatted_start_time": f"{int(chunk['start_time']//3600):02d}:{int((chunk['start_time']%3600)//60):02d}:{int(chunk['start_time']%60):02d}",
                "formatted_end_time": f"{int(chunk['end_time']//3600):02d}:{int((chunk['end_time']%3600)//60):02d}:{int(chunk['end_time']%60):02d}",
                "text": chunk["text"],
                "chapter_id": chapter_id
            }
            metadatas.append(metadata)
        
        # 8. Add to vector DB
        add_response = requests.post(VECTOR_DB_URL, json={"embeddings": embeddings, "metadatas": metadatas})
        add_response.raise_for_status()
        
        return {"message": "Audiobook processed and stored successfully", "audiobook_id": audiobook_id, "chunks": len(chunks), "chapters": len(chapters), "entities": len(entities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"service": "Pipeline Service", "status": "healthy"}
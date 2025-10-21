from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import os

app = FastAPI(title="Metadata DB Service", description="Manage audiobook metadata with SQLite")

DB_PATH = "audiobook_metadata.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    with open("schema.sql", "r") as f:
        schema = f.read()
    cursor.executescript(schema)
    conn.commit()
    conn.close()

# Init DB on startup
if not os.path.exists(DB_PATH):
    init_db()

class AudiobookCreate(BaseModel):
    title: str
    author: Optional[str] = None
    duration: Optional[float] = None

class ChapterCreate(BaseModel):
    audiobook_id: int
    title: str
    start_time: float
    end_time: float
    summary: Optional[str] = None

class ChunkCreate(BaseModel):
    audiobook_id: int
    chapter_id: Optional[int] = None
    start_time: float
    end_time: float
    text: str
    token_count: int
    embedding_id: int

class EntityCreate(BaseModel):
    name: str
    type: str
    audiobook_id: int

class EntityMentionCreate(BaseModel):
    entity_id: int
    chunk_id: int
    start_pos: int
    end_pos: int

@app.post("/audiobooks")
async def create_audiobook(audiobook: AudiobookCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO audiobooks (title, author, duration) VALUES (?, ?, ?)",
                   (audiobook.title, audiobook.author, audiobook.duration))
    audiobook_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"audiobook_id": audiobook_id}

@app.post("/chapters")
async def create_chapter(chapter: ChapterCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chapters (audiobook_id, title, start_time, end_time, summary) VALUES (?, ?, ?, ?, ?)",
                   (chapter.audiobook_id, chapter.title, chapter.start_time, chapter.end_time, chapter.summary))
    chapter_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"chapter_id": chapter_id}

@app.post("/chunks")
async def create_chunk(chunk: ChunkCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chunks (audiobook_id, chapter_id, start_time, end_time, text, token_count, embedding_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (chunk.audiobook_id, chunk.chapter_id, chunk.start_time, chunk.end_time, chunk.text, chunk.token_count, chunk.embedding_id))
    chunk_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"chunk_id": chunk_id}

@app.post("/entities")
async def create_entity(entity: EntityCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO entities (name, type, audiobook_id) VALUES (?, ?, ?)",
                   (entity.name, entity.type, entity.audiobook_id))
    entity_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"entity_id": entity_id}

@app.post("/entity_mentions")
async def create_entity_mention(mention: EntityMentionCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO entity_mentions (entity_id, chunk_id, start_pos, end_pos) VALUES (?, ?, ?, ?)",
                   (mention.entity_id, mention.chunk_id, mention.start_pos, mention.end_pos))
    mention_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"mention_id": mention_id}

@app.get("/audiobooks/{audiobook_id}/chapters")
async def get_chapters(audiobook_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chapters WHERE audiobook_id = ?", (audiobook_id,))
    chapters = cursor.fetchall()
    conn.close()
    return {"chapters": chapters}

@app.get("/chunks")
async def get_chunks(audiobook_id: int, chapter_id: Optional[int] = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if chapter_id:
        cursor.execute("SELECT * FROM chunks WHERE audiobook_id = ? AND chapter_id = ?", (audiobook_id, chapter_id))
    else:
        cursor.execute("SELECT * FROM chunks WHERE audiobook_id = ?", (audiobook_id,))
    chunks = cursor.fetchall()
    conn.close()
    return {"chunks": chunks}

@app.get("/entities/{audiobook_id}")
async def get_entities(audiobook_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM entities WHERE audiobook_id = ?", (audiobook_id,))
    entities = cursor.fetchall()
    conn.close()
    return {"entities": entities}

@app.get("/")
async def root():
    return {"service": "Metadata DB Service", "status": "healthy"}
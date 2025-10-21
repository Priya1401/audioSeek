-- Audiobook Metadata Database Schema

CREATE TABLE audiobooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author TEXT,
    duration REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audiobook_id INTEGER,
    title TEXT,
    start_time REAL,
    end_time REAL,
    summary TEXT,
    FOREIGN KEY (audiobook_id) REFERENCES audiobooks(id)
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audiobook_id INTEGER,
    chapter_id INTEGER,
    start_time REAL,
    end_time REAL,
    text TEXT NOT NULL,
    token_count INTEGER,
    embedding_id INTEGER,  -- reference to FAISS index
    FOREIGN KEY (audiobook_id) REFERENCES audiobooks(id),
    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
);

CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT,  -- character, place, etc.
    audiobook_id INTEGER,
    FOREIGN KEY (audiobook_id) REFERENCES audiobooks(id)
);

CREATE TABLE entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER,
    chunk_id INTEGER,
    start_pos INTEGER,
    end_pos INTEGER,
    FOREIGN KEY (entity_id) REFERENCES entities(id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

-- Indexes for performance
CREATE INDEX idx_chunks_audiobook ON chunks(audiobook_id);
CREATE INDEX idx_chunks_chapter ON chunks(chapter_id);
CREATE INDEX idx_entities_audiobook ON entities(audiobook_id);
-- Audiobook Metadata Database Schema (Book-Aware Version)

-- ================================
-- AUDIOBOOKS TABLE (one row per book)
-- ================================
CREATE TABLE IF NOT EXISTS audiobooks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    duration REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================
-- CHAPTERS TABLE (separated by book)
-- ================================
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    chapter_number INTEGER,
    title TEXT,
    start_time REAL,
    end_time REAL,
    summary TEXT,
    FOREIGN KEY (book_id) REFERENCES audiobooks(book_id)
);

-- ================================
-- CHUNKS TABLE (book-isolated)
-- ================================
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    chapter_id INTEGER,
    start_time REAL,
    end_time REAL,
    text TEXT NOT NULL,
    token_count INTEGER,
    embedding_id INTEGER,
    source_file TEXT,
    FOREIGN KEY (book_id) REFERENCES audiobooks(book_id),
    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
);

-- ================================
-- ENTITIES TABLE (per book)
-- ================================
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (book_id) REFERENCES audiobooks(book_id)
);

-- ================================
-- ENTITY MENTIONS (chunk references)
-- ================================
CREATE TABLE IF NOT EXISTS entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER,
    chunk_id INTEGER,
    start_pos INTEGER,
    end_pos INTEGER,
    FOREIGN KEY (entity_id) REFERENCES entities(id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

-- ================================
-- PERFORMANCE INDEXES
-- ================================
CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter_id);
CREATE INDEX IF NOT EXISTS idx_entities_book ON entities(book_id);
CREATE INDEX IF NOT EXISTS idx_chapters_book ON chapters(book_id);

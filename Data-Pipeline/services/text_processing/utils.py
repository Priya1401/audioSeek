import os
import re
from typing import List, Dict, Any, Optional

import spacy
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = embedding_model.tokenizer

# Load models once
nlp = spacy.load("en_core_web_trf")


def parse_transcript(transcript: str) -> List[Dict[str, Any]]:
    """Parse transcript text into segments with timestamps"""
    lines = transcript.strip().split('\n')
    segments = []
    for line in lines:
        match = re.match(r'\[(\d+\.\d+)-(\d+\.\d+)\]\s*(.*)', line)
        if match:
            start, end, text = match.groups()
            segments.append({
                'start': float(start),
                'end': float(end),
                'text': text.strip()
            })
    return segments


def extract_chapter_from_filename(filename: str) -> Optional[int]:
    """
    Extract chapter number from filename.
    Examples:
      - audiobook_romeo_and_juliet_chapter_04.txt -> 4
      - audiobook_edison_lifeinventions_chapter_02.txt -> 2
    """
    match = re.search(r'chapter[_\s](\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def detect_chapters(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect chapter markers in transcript segments"""
    chapters = []
    current_chapter = None

    for seg in segments:
        text = seg['text']

        # Match both "Chapter 1" and "CHAPTER I" (Roman numerals)
        match = re.search(r'chapter\s+([IVXLCDM]+|\d+)', text, re.IGNORECASE)

        if match:
            chapter_identifier = match.group(1)

            if chapter_identifier.isdigit():
                chapter_num = int(chapter_identifier)
            else:
                chapter_num = roman_to_int(chapter_identifier)

            # Close the previous chapter
            if current_chapter:
                current_chapter['end_time'] = seg['start']
                chapters.append(current_chapter)

            # Start new chapter
            current_chapter = {
                'id': chapter_num,
                'title': match.group(0),
                'start_time': seg['start']
            }

    # Close the last chapter
    if current_chapter:
        current_chapter['end_time'] = segments[-1]['end']
        chapters.append(current_chapter)

    return chapters


def roman_to_int(s: str) -> int:
    """Convert Roman numeral to integer"""
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }

    s = s.upper()
    total = 0
    prev_value = 0

    for char in reversed(s):
        value = roman_values.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract named entities from text using spaCy"""
    doc = nlp(text)
    entities = [{'name': ent.text, 'type': ent.label_} for ent in doc.ents]
    return entities


def chunk_text(
    segments: List[Dict[str, Any]],
    target_tokens: int,
    overlap_tokens: int,
    chapters: List[Dict[str, Any]],
    fallback_chapter_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Chunk transcript segments into token-sized chunks with overlap.

    Args:
        segments: List of transcript segments
        target_tokens: Target token count per chunk
        overlap_tokens: Number of overlapping tokens between chunks
        chapters: List of detected chapters from content
        fallback_chapter_id: Chapter ID from filename if no chapters detected in content
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for seg in segments:
        # Reset chapter_id for each segment
        chapter_id = None

        # First, try to find chapter from detected chapters
        for chap in chapters:
            if chap['start_time'] <= seg['start'] < chap['end_time']:
                chapter_id = chap['id']
                break

        # If no chapter found and we have a fallback, use it
        if chapter_id is None and fallback_chapter_id is not None:
            chapter_id = fallback_chapter_id

        tokens = len(tokenizer.encode(seg['text']))

        if current_tokens + tokens > target_tokens and current_chunk:
            chunk_text = ' '.join([s['text'] for s in current_chunk])
            entities = extract_entities(chunk_text)

            # Determine chapter_id for the chunk
            chunk_chapter_id = None
            first_seg_start = current_chunk[0]['start']

            # Try detected chapters first
            for chap in chapters:
                if chap['start_time'] <= first_seg_start < chap['end_time']:
                    chunk_chapter_id = chap['id']
                    break

            # Use fallback if no chapter found
            if chunk_chapter_id is None and fallback_chapter_id is not None:
                chunk_chapter_id = fallback_chapter_id

            chunks.append({
                'start_time': current_chunk[0]['start'],
                'end_time': current_chunk[-1]['end'],
                'text': chunk_text,
                'token_count': current_tokens,
                'chapter_id': chunk_chapter_id,
                'entities': entities
            })

            overlap_segments = max(1, overlap_tokens // 50)
            current_chunk = current_chunk[-overlap_segments:]
            current_tokens = sum(
                len(tokenizer.encode(s['text'])) for s in current_chunk)

        current_chunk.append(seg)
        current_tokens += tokens

    # Handle the last chunk
    if current_chunk:
        chunk_text = ' '.join([s['text'] for s in current_chunk])
        entities = extract_entities(chunk_text)

        # Determine chapter_id for final chunk
        chunk_chapter_id = None
        first_seg_start = current_chunk[0]['start']

        for chap in chapters:
            if chap['start_time'] <= first_seg_start < chap['end_time']:
                chunk_chapter_id = chap['id']
                break

        if chunk_chapter_id is None and fallback_chapter_id is not None:
            chunk_chapter_id = fallback_chapter_id

        chunks.append({
            'start_time': current_chunk[0]['start'],
            'end_time': current_chunk[-1]['end'],
            'text': chunk_text,
            'token_count': current_tokens,
            'chapter_id': chunk_chapter_id,
            'entities': entities
        })

    return chunks


def collect_unique_entities(chunks: List[Dict[str, Any]]) -> List[
    Dict[str, str]]:
    """Collect unique entities from all chunks"""
    all_entities = []
    seen = set()
    for chunk in chunks:
        for ent in chunk['entities']:
            key = (ent['name'], ent['type'])
            if key not in seen:
                all_entities.append(ent)
                seen.add(key)
    return all_entities


def extract_book_id_from_path(
    book_id: Optional[str] = None,
    folder_path: Optional[str] = None,
    file_path: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    chunks_file: Optional[str] = None
) -> str:
    """
    Extract book_id from various path sources.
    Priority: book_id > folder_path > file_path > file_paths > chunks_file > 'default'
    """
    if book_id:
        return book_id

    if folder_path:
        # Extract from folder: "/path/to/romeo_and_juliet" -> "romeo_and_juliet"
        return os.path.basename(os.path.normpath(folder_path))

    if file_path:
        # Extract from parent folder of file
        return os.path.basename(os.path.dirname(file_path))

    if file_paths and len(file_paths) > 0:
        # Extract from first file's parent folder
        return os.path.basename(os.path.dirname(file_paths[0]))

    if chunks_file:
        # Try to extract from chunks filename pattern
        # e.g., "romeo_and_juliet_chunks.json" -> "romeo_and_juliet"
        basename = os.path.basename(chunks_file)
        if '_chunks' in basename:
            return basename.split('_chunks')[0]
        # Or extract from parent folder
        parent = os.path.dirname(chunks_file)
        if parent and parent != '.':
            return os.path.basename(parent)

    # logger.warning("Could not extract book_id, using 'default'")
    return "default"

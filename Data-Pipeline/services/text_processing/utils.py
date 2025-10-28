import re
from typing import List, Dict, Any

import spacy
from transformers import AutoTokenizer

# Load models once
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def parse_transcript(transcript: str) -> List[Dict[str, Any]]:
    """Parse transcript text into segments with timestamps"""
    # Assume it'll be like [14.12-20.24]  Recording by Colleen McMahon. Historical Mysteries by Andrew Lang.
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


def detect_chapters(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect chapter markers in transcript segments"""
    chapters = []
    current_chapter = None
    for seg in segments:
        text = seg['text'].lower()
        match = re.search(r'chapter (\d+)', text)
        if match:
            if current_chapter:
                current_chapter['end_time'] = seg['start']
                chapters.append(current_chapter)
            current_chapter = {
                'id': int(match.group(1)),
                'title': f"Chapter {match.group(1)}",
                'start_time': seg['start']
            }
    if current_chapter:
        current_chapter['end_time'] = segments[-1]['end']
        chapters.append(current_chapter)
    return chapters


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract named entities from text using spaCy"""
    doc = nlp(text)
    entities = [{'name': ent.text, 'type': ent.label_} for ent in doc.ents]
    return entities


def chunk_text(
        segments: List[Dict[str, Any]],
        target_tokens: int,
        overlap_tokens: int,
        chapters: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Chunk transcript segments into token-sized chunks with overlap"""
    chunks = []
    current_chunk = []
    current_tokens = 0
    chapter_id = None

    for seg in segments:
        for chap in chapters:
            if chap['start_time'] <= seg['start'] < chap['end_time']:
                chapter_id = chap['id']
                break

        tokens = len(tokenizer.encode(seg['text']))

        if current_tokens + tokens > target_tokens and current_chunk:
            chunk_text = ' '.join([s['text'] for s in current_chunk])
            entities = extract_entities(chunk_text)
            chunks.append({
                'start_time': current_chunk[0]['start'],
                'end_time': current_chunk[-1]['end'],
                'text': chunk_text,
                'token_count': current_tokens,
                'chapter_id': chapter_id,
                'entities': entities
            })

            overlap_segments = max(1, overlap_tokens // 50)
            current_chunk = current_chunk[-overlap_segments:]
            current_tokens = sum(len(tokenizer.encode(s['text'])) for s in current_chunk)

        current_chunk.append(seg)
        current_tokens += tokens

    if current_chunk:
        chunk_text = ' '.join([s['text'] for s in current_chunk])
        entities = extract_entities(chunk_text)
        chunks.append({
            'start_time': current_chunk[0]['start'],
            'end_time': current_chunk[-1]['end'],
            'text': chunk_text,
            'token_count': current_tokens,
            'chapter_id': chapter_id,
            'entities': entities
        })

    return chunks


def collect_unique_entities(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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

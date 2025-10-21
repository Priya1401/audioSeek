from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import re
import spacy
from transformers import AutoTokenizer
import os
import json

app = FastAPI(title="Chunking Service", description="Chunk transcripts with chapter detection and metadata")

# Load models (assume en_core_web_sm is downloaded)
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

class ChunkingRequest(BaseModel):
    file_path: str
    target_tokens: int = 512
    overlap_tokens: int = 100
    output_file: str = None  # Optional: if provided, saves to file

class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    output_file: str = None

def parse_transcript(transcript: str):
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

def detect_chapters(segments):
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

def extract_entities(text: str):
    doc = nlp(text)
    entities = [{'name': ent.text, 'type': ent.label_} for ent in doc.ents]
    return entities

def chunk_text(segments, target_tokens, overlap_tokens, chapters):
    chunks = []
    current_chunk = []
    current_tokens = 0
    chapter_id = None
    for seg in segments:
        # Assign chapter
        for chap in chapters:
            if chap['start_time'] <= seg['start'] < chap['end_time']:
                chapter_id = chap['id']
                break
        tokens = len(tokenizer.encode(seg['text']))
        if current_tokens + tokens > target_tokens and current_chunk:
            # Create chunk
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
            # Overlap: keep last few segments
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

@app.post("/chunk", response_model=ChunkResponse)
async def chunk_transcript(request: ChunkingRequest):
    # Check if file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    # Check if it's a txt file
    if not request.file_path.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    
    # Read file content
    try:
        with open(request.file_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    segments = parse_transcript(transcript)
    
    if not segments:
        raise HTTPException(status_code=400, detail="No valid transcript segments found in file")
    
    chapters = detect_chapters(segments)
    chunks = chunk_text(segments, request.target_tokens, request.overlap_tokens, chapters)
    
    # Collect unique entities
    all_entities = []
    seen = set()
    for chunk in chunks:
        for ent in chunk['entities']:
            key = (ent['name'], ent['type'])
            if key not in seen:
                all_entities.append(ent)
                seen.add(key)
    
    response_data = {
        'chunks': chunks,
        'chapters': chapters,
        'entities': all_entities
    }
    
    # Save to file if output_file is provided
    if request.output_file:
        try:
            with open(request.output_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2)
            response_data['output_file'] = request.output_file
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving output file: {str(e)}")
    
    return ChunkResponse(**response_data)

@app.get("/")
async def root():
    return {"service": "Chunking Service", "status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
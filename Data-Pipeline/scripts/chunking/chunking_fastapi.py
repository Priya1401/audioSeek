import re
from datetime import timedelta
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

app = FastAPI(
    title="Transcript Chunking Service",
    description="API for processing and chunking audio transcripts",
    version="1.0.0"
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChunkingConfig(BaseModel):
    target_tokens: int = Field(default=512, ge=100, le=2048, description="Target tokens per chunk")
    overlap_tokens: int = Field(default=100, ge=0, le=500, description="Overlap tokens between chunks")
    warn_oversized: bool = Field(default=True, description="Warn about oversized segments")


class ProcessTranscriptRequest(BaseModel):
    transcript_content: str = Field(..., description="Raw transcript content with timestamps")
    config: Optional[ChunkingConfig] = Field(default_factory=ChunkingConfig)


class SegmentResponse(BaseModel):
    segment_id: int
    start_time: str
    end_time: str
    duration: float
    formatted_start: str
    formatted_end: str
    text: str


class ChunkResponse(BaseModel):
    chunk_id: int
    start_time: str
    end_time: str
    duration: float
    formatted_start_time: str
    formatted_end_time: str
    token_count: float
    segment_count: int
    segment_ids: List[int]
    segments: List[SegmentResponse]


class ProcessResponse(BaseModel):
    total_chunks: int
    total_segments: int
    total_duration: float
    avg_chunk_size: float
    avg_chunk_duration: float
    oversized_count: int
    max_segment_tokens: float
    chunks: List[ChunkResponse]


class TimeRangeQuery(BaseModel):
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class KeywordQuery(BaseModel):
    keyword: str = Field(..., min_length=1, description="Keyword to search for")


# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def parse_transcript(content: str) -> List[dict]:
    """Convert raw Whisper transcript into structured segments."""
    segments = []
    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r'\[([\d.]+)-([\d.]+)]\s*(.+)', line)
        if not match:
            continue

        start_time = match.group(1)
        end_time = match.group(2)
        text = match.group(3)

        segment = {
            'segment_id': len(segments),
            'start_time': start_time,
            'end_time': end_time,
            'duration': round(float(end_time) - float(start_time), 2),
            'text': text
        }
        segments.append(segment)

    return segments


def format_timestamps(seconds: float) -> str:
    """Convert seconds timestamp into HH:MM:SS or MM:SS format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(round(td.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    else:
        return f'{minutes:02d}:{seconds:02d}'


def update_timestamps(segments: List[dict]) -> List[dict]:
    """Add formatted timestamps to each segment."""
    for segment in segments:
        segment['formatted_start'] = format_timestamps(float(segment['start_time']))
        segment['formatted_end'] = format_timestamps(float(segment['end_time']))
    return segments


def count_token(text: str) -> float:
    """Estimate token count based on word count."""
    words = text.split()
    return len(words) * 1.33


def create_chunk_dict(segments: List[dict], chunk_id: int) -> dict:
    """Create a chunk dictionary from segments."""
    combined_text = ' '.join([seg['text'] for seg in segments])

    start_time = segments[0]['start_time']
    end_time = segments[-1]['end_time']
    duration = float(segments[-1]['end_time']) - float(segments[0]['start_time'])

    formatted_start_time = segments[0]['formatted_start']
    formatted_end_time = segments[-1]['formatted_end']

    chunk = {
        'chunk_id': chunk_id,
        'start_time': start_time,
        'end_time': end_time,
        'duration': round(duration, 2),
        'formatted_start_time': formatted_start_time,
        'formatted_end_time': formatted_end_time,
        'token_count': count_token(combined_text),
        'segment_count': len(segments),
        'segment_ids': [seg['segment_id'] for seg in segments],
        'segments': segments
    }

    return chunk


def create_chunks(segments: List[dict], target_tokens: int = 512,
                  overlap_tokens: int = 100, warn_oversized: bool = True) -> tuple:
    """Create overlapping chunks from segments."""
    oversized_segments = []
    max_segment_tokens = 0

    chunks = []
    current_chunk_segments = []
    current_tokens = 0
    chunk_id = 0

    i = 0
    while i < len(segments):
        segment = segments[i]
        segment_tokens = count_token(segment['text'])

        if segment_tokens > max_segment_tokens:
            max_segment_tokens = segment_tokens

        if segment_tokens > target_tokens:
            oversized_segments.append({
                'segment_id': segment.get('segment_id', i),
                'tokens': segment_tokens,
                'excess': segment_tokens - target_tokens,
                'excess_pct': (segment_tokens - target_tokens) / target_tokens * 100,
                'time': segment.get('formatted_start', 'Unknown')
            })

        if current_tokens + segment_tokens > target_tokens and current_chunk_segments:
            chunk = create_chunk_dict(current_chunk_segments, chunk_id)
            chunks.append(chunk)
            chunk_id += 1

            overlap_segments = []
            overlap_token_count = 0

            for seg in reversed(current_chunk_segments):
                seg_tokens = count_token(seg['text'])
                if overlap_token_count + seg_tokens <= overlap_tokens:
                    overlap_segments.insert(0, seg)
                    overlap_token_count += seg_tokens
                else:
                    break

            current_chunk_segments = overlap_segments
            current_tokens = overlap_token_count

        current_chunk_segments.append(segment)
        current_tokens += segment_tokens
        i += 1

    if current_chunk_segments:
        chunk = create_chunk_dict(current_chunk_segments, chunk_id)
        chunks.append(chunk)

    return chunks, oversized_segments, max_segment_tokens


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Transcript Chunking Service",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/process", response_model=ProcessResponse)
async def process_transcript(request: ProcessTranscriptRequest):
    """
    Process a transcript and return chunks.
    
    The transcript should be in the format:
    [start_time-end_time] text
    
    Example:
    [0.0-5.2] Hello everyone, welcome to the audiobook.
    [5.2-10.5] Today we'll be discussing machine learning.
    """
    try:
        # Parse transcript
        segments = parse_transcript(request.transcript_content)

        if not segments:
            raise HTTPException(
                status_code=400,
                detail="No valid segments found in transcript. Check format: [start-end] text"
            )

        # Update timestamps
        segments = update_timestamps(segments)

        # Create chunks
        chunks, oversized_segments, max_segment_tokens = create_chunks(
            segments,
            target_tokens=request.config.target_tokens,
            overlap_tokens=request.config.overlap_tokens,
            warn_oversized=request.config.warn_oversized
        )

        # Calculate statistics
        total_duration = float(segments[-1]['end_time']) - float(segments[0]['start_time'])
        avg_chunk_size = sum(c['token_count'] for c in chunks) / len(chunks) if chunks else 0
        avg_chunk_duration = sum(c['duration'] for c in chunks) / len(chunks) if chunks else 0

        return ProcessResponse(
            total_chunks=len(chunks),
            total_segments=len(segments),
            total_duration=round(total_duration, 2),
            avg_chunk_size=round(avg_chunk_size, 2),
            avg_chunk_duration=round(avg_chunk_duration, 2),
            oversized_count=len(oversized_segments),
            max_segment_tokens=round(max_segment_tokens, 2),
            chunks=chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/process/file")
async def process_transcript_file(
        file: UploadFile = File(...),
        target_tokens: int = 512,
        overlap_tokens: int = 100,
        warn_oversized: bool = True
):
    """
    Process a transcript file and return chunks.
    
    Upload a .txt file with transcript in the format:
    [start_time-end_time] text
    """
    try:
        # Read file content
        content = await file.read()
        transcript_content = content.decode('utf-8')

        # Create request object
        config = ChunkingConfig(
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            warn_oversized=warn_oversized
        )

        request = ProcessTranscriptRequest(
            transcript_content=transcript_content,
            config=config
        )

        # Process using existing endpoint logic
        return await process_transcript(request)

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@app.post("/search/time_range")
async def search_time_range(
        query: TimeRangeQuery,
        request: ProcessTranscriptRequest
):
    """Search for chunks within a specific time range."""
    try:
        # Process transcript first
        result = await process_transcript(request)

        # Filter chunks by time range
        filtered_chunks = [
            chunk for chunk in result.chunks
            if float(chunk.start_time) <= query.end_time and float(chunk.end_time) >= query.start_time
        ]

        return {
            "query": {
                "start_time": query.start_time,
                "end_time": query.end_time,
                "formatted_range": f"{format_timestamps(query.start_time)} - {format_timestamps(query.end_time)}"
            },
            "results_count": len(filtered_chunks),
            "chunks": filtered_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/search/keyword")
async def search_keyword(
        query: KeywordQuery,
        request: ProcessTranscriptRequest
):
    """Search for chunks containing a specific keyword."""
    try:
        # Process transcript first
        result = await process_transcript(request)

        # Filter chunks by keyword
        keyword_lower = query.keyword.lower()
        filtered_chunks = [
            chunk for chunk in result.chunks
            if any(keyword_lower in seg.text.lower() for seg in chunk.segments)
        ]

        return {
            "query": {
                "keyword": query.keyword
            },
            "results_count": len(filtered_chunks),
            "chunks": filtered_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

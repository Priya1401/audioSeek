import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from faster_whisper import WhisperModel
import tempfile
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transcription Service", description="Transcribe audio to text with timestamps and chapter detection")

# Load model once
model = WhisperModel("base")

def detect_chapters_in_transcript(transcript):
    chapters = []
    lines = transcript.split('\n')
    for line in lines:
        match = re.search(r'chapter (\d+)', line, re.IGNORECASE)
        if match:
            time_match = re.match(r'\[(\d+\.\d+)-', line)
            start_time = float(time_match.group(1)) if time_match else 0
            chapters.append({
                'id': int(match.group(1)),
                'title': f"Chapter {match.group(1)}",
                'start_time': start_time
            })
    return chapters

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    allowed_exts = ('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma')
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    logger.info(f"Transcribing file: {file.filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        segments, info = model.transcribe(tmp_path)
        lines = [f"[{s.start:.2f}-{s.end:.2f}] {s.text}" for s in segments]
        transcript = "\n".join(lines)
        chapters = detect_chapters_in_transcript(transcript)
        logger.info(f"Transcription completed, {len(segments)} segments, {len(chapters)} chapters detected")
        return {
            "transcript": transcript,
            "language": getattr(info, 'language', 'unknown'),
            "duration": getattr(info, 'duration', None),
            "chapters": chapters
        }
    finally:
        os.unlink(tmp_path)

@app.get("/")
async def root():
    return {"service": "Transcription Service", "status": "healthy"}

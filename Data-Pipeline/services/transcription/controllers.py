from fastapi import APIRouter, UploadFile, File, HTTPException
from services import TranscriptionService
from models import TranscriptionResponse

router = APIRouter()
transcription_service = TranscriptionService()

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to text with timestamps and chapter detection"""
    try:
        return await transcription_service.transcribe_audio(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@router.get("/")
async def root():
    return {"service": "Transcription Service", "status": "healthy"}
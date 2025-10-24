from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    duration: Optional[float] = None
    chapters: List[Dict[str, Any]]
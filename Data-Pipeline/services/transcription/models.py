from typing import List, Dict, Any, Optional

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    transcript: str
    language: str
    duration: Optional[float] = None
    chapters: List[Dict[str, Any]]

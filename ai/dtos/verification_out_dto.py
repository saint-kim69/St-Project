from datetime import datetime

from pydantic import BaseModel


class VerificationOutDTO(BaseModel):
    started_at: datetime
    ended_at: datetime
    score: float
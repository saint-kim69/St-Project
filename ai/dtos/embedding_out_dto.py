from datetime import datetime

from pydantic import BaseModel

class EmbeddingOutDTO(BaseModel):
    started_at: datetime
    ended_at: datetime
    embedding: list[int]
    count: int
    
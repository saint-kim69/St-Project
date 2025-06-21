from datetime import datetime

from pydantic import BaseModel


class FindAnimalLog(BaseModel):
    animal_id: int
    image_id: int
    score: float
    origin_embedding_id: str
    compare_embedding_id: str
    created_at: datetime
    started_at: datetime
    ended_at: datetime
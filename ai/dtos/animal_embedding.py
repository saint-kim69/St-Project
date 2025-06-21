from datetime import datetime

from pydantic import BaseModel


class AnimalEmbedding(BaseModel):
    face: list[float]
    nose: list[float]
    eye_0: list[float]
    eye_1: list[float]



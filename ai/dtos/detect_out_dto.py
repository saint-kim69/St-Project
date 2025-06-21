from pydantic import BaseModel
from typing import Any

class MetaModel(BaseModel):
    x: float
    y: float
    w: float
    h: float
    confidence: float

class DetectModel(BaseModel):
    img: Any
    meta: MetaModel


class DetectOutDto(BaseModel):
    face: DetectModel
    nose: DetectModel
    eye_0: DetectModel
    eye_1: DetectModel
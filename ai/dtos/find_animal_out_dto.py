from pydantic import BaseModel


class FindAnimalOutDTO(BaseModel):
    name: str
    gender: str
    is_neutering: bool
    is_lost: bool
    is_inouculation: bool
    kind: str
    registration_no: str
    image: str
    compare_image: str
    score: float
    
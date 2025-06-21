from datetime import datetime
import requests

from ai.dtos.embedding_out_dto import EmbeddingOutDTO
from ai.dtos.verification_out_dto import VerificationOutDTO

class AIRequestService:
    def __init__(self, address=''):
        self.address = address

    def request_emedding(self, image) -> EmbeddingOutDTO:
        started_at = datetime.now()
        embedding = self.request_ai_server(image)
        ended_at = datetime.now()

        return EmbeddingOutDTO(
            started_at,
            ended_at,
            embedding,
            len(embedding)
        )
    
    def request_verification(self, embedding, compare_embedding) -> VerificationOutDTO:
        started_at = datetime.now()
        score = self._request_verification(embedding, compare_embedding)
        ended_at = datetime.now()
        return VerificationOutDTO(
            started_at,
            ended_at,
            score
        )
    
    def _request_embedding(self, image):
        pass

    def _request_verification(self, embedding, compare_embedding):
        pass
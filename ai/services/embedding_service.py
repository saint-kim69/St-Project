from datetime import datetime

from ai.dto.embedding_out_dto import EmbeddingOutDto

class EmbeddingService:

    def run(self, image) -> EmbeddingOutDto:
        started_at = datetime.now()
        embedding = self.request_ai_server(image)
        ended_at = datetime.now()

        return EmbeddingOutDto(
            started_at,
            ended_at,
            embedding,
            len(embedding)
        )

    def request_ai_server(image) -> list[int]:
        pass

        
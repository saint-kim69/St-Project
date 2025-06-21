from datetime import datetime 

from ai.models.animal import Animal
from ai.models.animal_image import AnimalImage
from ai.services.ai_request_service import AIRequestService
from ai.services.mongodb_manager import MongoDBManager
from ai.dtos.find_animal_log import FindAnimalLog
from ai.dtos.animal_embedding import AnimalEmbedding
from ai.dtos.find_animal_out_dto import FindAnimalOutDTO

class FindAnimalService:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.ai_request_service = AIRequestService()

    def run(self, image, purpose: str | None =None) -> FindAnimalOutDTO:
        embedding_started_at = datetime.now()
        embedding_out_dto = self.ai_request_service.request_emedding(image)
        embedding_ended_at = datetime.now()

        with MongoDBManager('ai', 'animal_image') as mongo:
            animal_image_embeddings = mongo.filter()
        
        find_started_at = datetime.now()
        for animal_image_embedding in animal_image_embeddings:
            verification_out_dto = self.ai_request_service.request_verification(
                animal_image_embedding.embedding, embedding_out_dto.embedding
                )
            if verification_out_dto.score > self.threshold:
                break
        else:
            raise Exception('찾을 수 없습니다.')
        find_ended_at = datetime.now()

        animal = Animal.objects.get(id=animal_image_embedding.animal_id)
        animal_image = AnimalImage.objects.get(id=animal_image_embedding.image_id)
        
        with MongoDBManager('ai', 'animal_image') as mongo:
            compare_animal_image = AnimalImage.objects.create(animal_id=animal_image_embedding.animal_id, image=image)
            compare_animal_image_embedding = AnimalEmbedding(
                animal_id=animal_image_embedding.animal_id, 
                image_id=compare_animal_image.id, 
                embedding=embedding_out_dto.embedding, 
                created_at=datetime.now(),
                started_at=embedding_started_at,
                ended_at=embedding_ended_at
            )
            compare_animal_image_embedding = mongo.save(compare_animal_image_embedding)


        with MongoDBManager('ai', 'find_animal_log') as mongo:
            find_animal_log = FindAnimalLog(
                animal_id=animal_image_embedding.animal_id,
                image_id=animal_image_embedding.image_id,
                score=verification_out_dto.score,
                origin_embedding_id=animal_image_embedding.id,
                compare_embedding_id=compare_animal_image_embedding.id,
                created_at=datetime.now(),
                started_at=find_started_at,
                ended_at=find_ended_at
            )
            mongo.save([find_animal_log])

        return FindAnimalOutDTO(
            name=animal.name,
            gender=animal.gender,
            is_neutering=animal.is_neutering,
            is_lost=animal.is_lost,
            is_inouculation=animal.is_inoculation,
            kind=animal.kind,
            registration_no=animal.registration_no,
            image=animal_image.image,
            compare_image=compare_animal_image.image,
            score=verification_out_dto.score
        )
from datetime import datetime

from ai.models import AnimalImage
from ai.services.ai_request_service import AIRequestService
from ai.dtos.animal_embedding import AnimalEmbedding
from demonstration.ai.services.mongodb_manager import MongoDBManager

"""
    1. embedding 검출 순서
    image 에서 동물의 얼굴을 검출하고 거기에서 중요한 것이 score 가 어느정도 높은 경우에 대해서만 대응하도록 적용
    image 에서 동물의 얼굴이 여러개인 경우에는 반환을 에러로 하고 해당하는 부분에 정확하게 하나의 동물만 인식해달라고 요청
    image 에서 동물의 얼굴이 없는 경우에는 에러를 반환을 하고 동물을 향해서 카메라를 적용해달라고 요청
    인식이 되어지는 경우 embedding 값을 동물의 id와 image를 박아서 mongodb에 저장, 해당하는 방식의 경우 추후에 이미지 퀄리티를 선택해서 할 수 있는 형태로 제공필요
    embedding 의 경우 최소 5장을 만드는 것을 목표로 하기에 숫자를 반환하는 것으로 한다

    2. verification 적용 순서
    두개다 embedding 값이 있다고 판단(origin, compare)
    origin 의 경우 DB 에 저장이 되어진 embedding
    compare 의 경우 비교하기 위해 들어온 image에서 뽑아온 embedding
    중요한 점 origin 의 경우 여러개가 존재하는 것(등록된 동물이 많을 수록 늘어남)-어떻게 가져오냐가 관건, 특징을 입력받도록 하는 방식에 더해서 추가적으로 필요한 방법을 강구
    둘이 비교한 부분을 threshold 설정을 통해서 동일한지에 대해 면밀히 검토하여 인증여부 판단
"""

class DetectService:

    def __init__(self):
        self.ai_request_service = AIRequestService()

    def run(self, image, animal_id) -> int:
        animal_image = AnimalImage.objects.create(
            animal_id=animal_id,
            image=image
        )

        embedding_out_dto = self.ai_request_service.request_emedding(image)        
        if embedding_out_dto.count != 1:
            raise Exception(f'현재 인식된 둥물이 {embedding_out_dto.count} 입니다. 하나의 동물을 인식시켜주세요.')

        animal_embedding = AnimalEmbedding(
            animal_id,
            animal_image.id,
            embedding_out_dto.embedding,
            datetime.now(),
            embedding_out_dto.started_at,
            embedding_out_dto.ended_at
        )
        with MongoDBManager('ai', 'animal_image') as mongo:
            mongo.save(animal_embedding.model_dump_json())

        return AnimalImage.objects.filter(animal_id=animal_id).count()

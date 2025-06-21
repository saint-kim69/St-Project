from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from channels.generic.websocket import WebsocketConsumer
from PIL import Image
from uuid import uuid4
import os
from django.conf import settings
from ai.utils.detect_objects import detect_image
import json
from asgiref.sync import async_to_sync
import base64
import numpy as np
import cv2
from ai.utils.embedding_object import ImageEncoder, embedding
from django.db.models import Max
from django.core.files.base import ContentFile

class DetectConsumer(WebsocketConsumer):
    def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_animal_register_{self.room_name}'
        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name, self.channel_name
        )
        self.accept()

    def disconnect(self, code):
        async_to_sync(self.channel_layer.group_discard)(
            self.room_group_name, self.channel_name
        )

    def receive(self, text_data):
        from ai.models.animal_image import AnimalImage

        text_data_json = json.loads(text_data)
        if ',' in text_data_json['image']:
            base64_data = text_data_json['image'].split(',')[1]
        else:
            base64_data = text_data_json['image']
        img_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detect_result = detect_image(None, img)
        if detect_result is None:
            return
        result = AnimalImage.objects.filter(animal_id=self.room_name).aggregate(max_seq=Max('seq'))
        max_seq = result.get('max_seq') if result.get('max_seq') else 0
        animal_image = AnimalImage(
            animal_id=self.room_name,
            is_main=False,
            seq=max_seq+1,
            category='full',
        )
        keys = ['face', 'nose', 'eye_0', 'eye_1']
        for key in keys:
            encoder = ImageEncoder()
            embedding_result = encoder(getattr(getattr(detect_result, key), 'img'))[0]
            # embedding_result = embedding(None, getattr(getattr(detect_result, key), 'img'))
            if key == 'face':
                is_success, buffer = cv2.imencode('.jpg', getattr(getattr(detect_result, key), 'img'))
            setattr(animal_image, f'{key}_embedding', embedding_result.tolist())
            setattr(animal_image, f'{key}_meta', getattr(getattr(detect_result, key), 'meta').model_dump())
        animal_image.image.save(
            f'animal_id_{animal_image.animal_id}.jpg',
            ContentFile(buffer.tobytes()),
            save=True
        )
        async_to_sync(self.channel_layer.group_send)(
            self.room_group_name, {'type': 'seq', 'max_seq': max_seq}
        )

    def seq(self, event):
        message = event['max_seq']
        self.send(text_data=json.dumps({"message": message}))

    # def post(self, request):
    #     print('here')
    #     file_obj = request.data['file']
    #     print(file_obj)
    #     # file_path = os.path.join('static', f'./file{uuid4()}.jpg')
    #     # full_path = os.path.join(settings.MEDIA_ROOT, file_path)
    #     # directory = os.path.dirname(full_path)
    #     # print(file_obj)
    #     # if not os.path.exists(directory):
    #     #     os.makedirs(directory)

    #     # with open(full_path, 'wb+') as dest:
    #     #     for chunk in file_obj.chunks():
    #     #         dest.write(chunk)
        
    #     return Response(status=200, data={'uuid': uuid4()})
    
    # def get(self, request):
    #     image_name = request.query_params.get('image')
    #     image_path = os.path.join(settings.BASE_DIR, f'media/static/{image_name}')
    #     result = detect_image(image_path)
    #     print(result)

    #     return Response(status=200, data=result)
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
from ai.dtos.animal_embedding import AnimalEmbedding
from django.core.paginator import Paginator
from ai.utils.calc_similarity import calc_similarity


class FindAnimalConsumer(WebsocketConsumer):
    def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = f"find_animal_{self.room_name}"
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
        if "," in text_data_json["image"]:
            base64_data = text_data_json["image"].split(",")[1]
        else:
            base64_data = text_data_json["image"]
        img_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detect_result = detect_image(None, img)
        if detect_result is None:
            print("no detect")
            return

        keys = ["face", "nose", "eye_0", "eye_1"]
        embedding_results = {"face": None, "nose": None, "eye_0": None, "eye_1": None}
        for key in keys:
            encoder = ImageEncoder()
            embedding_result = encoder(getattr(getattr(detect_result, key), "img"))
            # embedding_result = embedding(None, getattr(getattr(detect_result, key), 'img'))
            embedding_results.update({key: embedding_result.tolist()})
        compare = AnimalEmbedding(
            face=embedding_results["face"],
            nose=embedding_results["nose"],
            eye_0=embedding_results["eye_0"],
            eye_1=embedding_results["eye_1"],
        )
        is_find = False

        paginator = Paginator(AnimalImage.objects.filter(), 10)
        animal_id = None
        for i in range(paginator.num_pages):
            if is_find:
                break
            page = paginator.get_page(i)
            for animal_image in page.object_list:
                origin = AnimalEmbedding(
                    face=animal_image.face_embedding,
                    nose=animal_image.nose_embedding,
                    eye_0=animal_image.eye_0_embedding,
                    eye_1=animal_image.eye_1_embedding,
                )
                score = calc_similarity(origin, compare)
                if score > 0.85:
                    is_find = True
                    animal_id = animal_image.animal_id
                    break

        if not is_find:
            print("no find")
            return
        if is_find:
            async_to_sync(self.channel_layer.group_send)(
                self.room_group_name,
                {"type": "find_animal", "animal_id": animal_id, "score": score},
            )

    def find_animal(self, event):
        from ai.models import Animal

        animal_id = event["animal_id"]
        score = event["score"]
        try:
            animal = Animal.objects.get(id=animal_id)
        except Animal.DoesNotExist:
            return

        self.send(
            text_data=json.dumps(
                {
                    "name": animal.name,
                    "kind": animal.kind,
                    "registrationNo": animal.registration_no,
                    "gender": animal.gender,
                    "isInoculation": animal.is_inoculation,
                    "isLost": animal.is_lost,
                    "isNeutering": animal.is_neutering,
                    "score": score,
                }
            )
        )

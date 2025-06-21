from django.core.management.base import BaseCommand
from django.conf import settings
import glob
import os
import cv2
from ai.utils.detect_objects import detect_image
from ai.utils.embedding_object import ImageEncoder, embedding
from ai.models import Animal, AnimalImage
from random import randrange
from uuid import uuid4
from django.core.files.base import ContentFile


class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        fail_images = []
        keys = ["face", "nose", "eye_0", "eye_1"]
        kinds = ["골든리트리버", "닥스훈트", "치와와", "말티즈", "비숑"]
        booleans = [True, False]
        genders = ["M", "F"]
        init_data_dir_path = os.path.join(settings.BASE_DIR, "training_data")
        images = glob.glob(init_data_dir_path + "/*.jpg")
        registered_image = []
        for idx, image in enumerate(images):
            animal_name = image.split("/")[-1].split("-")[0]
            if animal_name in registered_image:
                continue

            registered_image.append(animal_name)
            animal = Animal.objects.create(
                name=f"animal_{idx}",
                gender=genders[randrange(0, 1)],
                registration_no=str(uuid4())[:16],
                kind=kinds[randrange(0, 4)],
                is_neutering=booleans[randrange(0, 1)],
                is_lost=booleans[randrange(0, 1)],
                is_inoculation=booleans[randrange(0, 1)],
            )
            img = cv2.imread(image)
            try:
                detect_result = detect_image(None, img)
            except Exception:
                animal.delete()
                continue

            if detect_result is None:
                fail_images.append(image)
                continue

            animal_image = AnimalImage(
                animal_id=animal.id, is_main=True, seq=1, category="full"
            )

            for key in keys:
                encoder = ImageEncoder()
                embedding_result = encoder(getattr(getattr(detect_result, key), "img"))
                # embedding_result = embedding(None, getattr(getattr(detect_result, key), 'img'))
                if key == "face":
                    is_success, buffer = cv2.imencode(
                        ".jpg", getattr(getattr(detect_result, key), "img")
                    )
                setattr(animal_image, f"{key}_embedding", embedding_result.tolist())
                setattr(
                    animal_image,
                    f"{key}_meta",
                    getattr(getattr(detect_result, key), "meta").model_dump(),
                )

            animal_image.image.save(
                f"animal_id_{animal_image.animal_id}.jpg",
                ContentFile(buffer.tobytes()),
                save=True,
            )

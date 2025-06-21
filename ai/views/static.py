from django.shortcuts import render, redirect
from ai.models import Animal
from ai.utils.embedding_object import ImageEncoder
from django.conf import settings
import os
import cv2
from django.http import HttpResponse
from ai.utils.detect_objects import detect_image
from ai.models.animal_image import AnimalImage
from django.core.files.base import ContentFile


def home(request):
    return render(request, 'ai/home.html')

def register(request):
    if request.method == 'POST':
        data = request.POST
        animal = Animal.objects.create(
            name=data['name'],
            gender=data['gender'],
            kind=data['kind'],
            registration_no=data['registration_no'],
            is_neutering= True if data['is_neutering'] == 1 else False,
            is_lost= True if data['is_lost'] == 1 else False,
            is_inoculation= True if data['is_inoculation'] == 1 else False,
        )
        return redirect('register_image', pk=animal.id)

    return render(request, 'ai/register.html')

def find(request):
    return render(request, 'ai/find_animal.html')


def register_image(request, pk):
    return render(request, 'ai/register_image.html', {'animal_id': pk})

def verification_image(request):
    return render(request, 'ai/verification.html')

def embedding(request):
    image_path = os.path.join(settings.BASE_DIR, 'media/test.jpg')

    img = cv2.imread(image_path)

    detect_result = detect_image(None, img)
    if detect_result is None:
        return
    animal_image = AnimalImage(
        animal_id=1,
        is_main=False,
        seq=10,
        category='full',
    )
    keys = ['face', 'nose', 'eye_0', 'eye_1']
    for key in keys:
        encoder = ImageEncoder()
        embedding_result = encoder(getattr(getattr(detect_result, key), 'img'))
        if key == 'face':
            is_success, buffer = cv2.imencode('.jpg', getattr(getattr(detect_result, key), 'img'))
        setattr(animal_image, f'{key}_embedding', embedding_result[0].tolist())
        setattr(animal_image, f'{key}_meta', getattr(getattr(detect_result, key), 'meta').model_dump())
    animal_image.image.save(
        f'animal_id_{animal_image.animal_id}.jpg',
        ContentFile(buffer.tobytes()),
        save=True
    )
    
    return HttpResponse({'data': 1})
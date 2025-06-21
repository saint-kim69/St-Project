from django.urls import re_path

from .views.detect import DetectConsumer
from .views.find_animal import FindAnimalConsumer

websocket_urlpatterns = [
    re_path(r'ai/detect/(?P<room_name>\w+)$', DetectConsumer.as_asgi()),
    re_path(r'ai/find-animal/(?P<room_name>\w+)$', FindAnimalConsumer.as_asgi()),
]
from django.urls import path

from .views.static import find, register, home, register_image, embedding


urlpatterns = [
    path('find', find),
    path('register', register),
    path('register-image/<int:pk>', register_image, name='register_image'),
    path('home', home),
    path('embedding', embedding)
]

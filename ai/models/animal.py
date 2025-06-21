from django.db import models


class Animal(models.Model):
    class Gender(models.TextChoices):
        MALE = 'M', '수컷'
        FEMALE = 'F', '암컷'

    name = models.CharField('이름', max_length=64)
    registration_no = models.CharField('등록번호', max_length=64)
    gender = models.CharField('성별', choices=Gender.choices, max_length=8)
    kind = models.CharField('동물종', max_length=256)
    is_neutering = models.BooleanField('중성화여부', default=False)
    is_lost = models.BooleanField('분실여부', default=False)
    is_inoculation = models.BooleanField('접종여부', default=False)

    class Meta:
        verbose_name = '동물'
        verbose_name_plural = '동물'
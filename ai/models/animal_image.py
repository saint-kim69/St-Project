from django.db import models

class AnimalImage(models.Model):
    animal = models.ForeignKey('ai.Animal', on_delete=models.CASCADE, related_name='images', related_query_name='images')
    seq = models.IntegerField(default=0)
    is_main = models.BooleanField('메인여부', default=False)
    image = models.ImageField('이미지')
    category = models.CharField(max_length=32, null=True)
    face_embedding = models.JSONField(null=True)
    nose_embedding = models.JSONField(null=True)
    eye_0_embedding = models.JSONField(null=True)
    eye_1_embedding = models.JSONField(null=True)
    face_meta = models.JSONField(null=True)
    nose_meta = models.JSONField(null=True)
    eye_0_meta = models.JSONField(null=True)
    eye_1_meta = models.JSONField(null=True)

    class Meta:
        verbose_name = '동물 이미지'
        verbose_name_plural = '동물 이미지'
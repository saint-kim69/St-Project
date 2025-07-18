# Generated by Django 5.2 on 2025-04-24 06:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai', '0002_animalimage_category_animalimage_embedding_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='animalimage',
            name='embedding',
        ),
        migrations.RemoveField(
            model_name='animalimage',
            name='meta',
        ),
        migrations.AddField(
            model_name='animalimage',
            name='eye_0_embedding',
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name='animalimage',
            name='eye_0_meta',
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name='animalimage',
            name='eye_1_embedding',
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name='animalimage',
            name='eye_1_meta',
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name='animalimage',
            name='face_embedding',
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name='animalimage',
            name='face_meta',
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name='animalimage',
            name='nose_embedding',
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name='animalimage',
            name='nose_meta',
            field=models.JSONField(null=True),
        ),
    ]

# Generated by Django 5.2 on 2025-04-22 23:16

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Animal',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=64, verbose_name='이름')),
                ('registration_no', models.CharField(max_length=64, verbose_name='등록번호')),
                ('gender', models.CharField(choices=[('M', '수컷'), ('F', '암컷')], max_length=8, verbose_name='성별')),
                ('kind', models.CharField(max_length=256, verbose_name='동물종')),
                ('is_neutering', models.BooleanField(default=False, verbose_name='중성화여부')),
                ('is_lost', models.BooleanField(default=False, verbose_name='분실여부')),
                ('is_inoculation', models.BooleanField(default=False, verbose_name='접종여부')),
            ],
            options={
                'verbose_name': '동물',
                'verbose_name_plural': '동물',
            },
        ),
        migrations.CreateModel(
            name='AnimalImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_main', models.BooleanField(default=False, verbose_name='메인여부')),
                ('image', models.ImageField(upload_to='', verbose_name='이미지')),
                ('animal', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='images', to='ai.animal')),
            ],
            options={
                'verbose_name': '동물 이미지',
                'verbose_name_plural': '동물 이미지',
            },
        ),
    ]

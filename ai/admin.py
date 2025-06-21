from django.contrib import admin
from ai.models import Animal, AnimalImage

class AnimalImageInline(admin.StackedInline):
    model = AnimalImage

class AnimalAdmin(admin.ModelAdmin):
    list_display = ['name', 'gender', 'registration_no', 'kind', 'is_neutering', 'is_lost', 'is_inoculation', 'count_of_image']
    fields = ['name', 'gender', 'registration_no', 'kind', 'is_neutering', 'is_lost', 'is_inoculation']
    inlines = [AnimalImageInline]

    @admin.display(
        description='등록된 이미지 개수'            
    )
    def count_of_image(self, obj):
        return AnimalImage.objects.filter(animal_id=obj.id).count()

# Register your models here.
admin.site.register(Animal, AnimalAdmin)
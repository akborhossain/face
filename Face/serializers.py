from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()


class DBImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
    name = serializers.CharField(max_length=100) 

from django.test import TestCase

# Create your tests here.
import requests

url = "http://localhost:8000/api/face-auth/"
files = {'image': open("C:\Users\Akbar\Downloads\Image.jpg", 'rb')}
response = requests.post(url, files=files)

print(response.json())
from django.urls import path
from .views import *

urlpatterns = [
    path('face-auth/', FaceAuthAPIView.as_view(), name='face_auth_api'),
    path('add-to-db/', AddToDBAPIView.as_view(), name='add_to_db'),
]
from django.urls import path
from .views import FaceAuthAPIView

urlpatterns = [
    path('face-auth/', FaceAuthAPIView.as_view(), name='face_auth_api'),
]
from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import *
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from .Silent_Face_Anti_Spoofing.src.generate_patches import CropImage
from .Silent_Face_Anti_Spoofing.src.utility import parse_model_name
from .util import *
import os

DEVICE_ID = 0
MODEL_DIR = "D:/Python project/Django/FaceAPI/Face/Silent_Face_Anti_Spoofing/resources/anti_spoof_models"
DB_DIR = "D:/Python project/Django/FaceAPI/Face/db"

face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def resize_image_to_3_4(image):
    height, width, _ = image.shape
    desired_ratio = 3 / 4
    current_ratio = width / height
    if current_ratio > desired_ratio:
        new_width = int(height * desired_ratio)
        offset = (width - new_width) // 2
        image = image[:, offset:offset + new_width]
    elif current_ratio < desired_ratio:
        new_height = int(width / desired_ratio)
        offset = (height - new_height) // 2
        image = image[offset:offset + new_height, :]
    return image

def test_antispoof(image):
    model_test = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()
    image = resize_image_to_3_4(image)
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
    return np.argmax(prediction)

class FaceAuthAPIView(APIView):
    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            np_img = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            faces = face_app.get(img)
            results = []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                h = y2 - y1
                w = x2 - x1
                y1 = max(0, y1 - int(0.25 * h))
                y2 = min(img.shape[0], y2 + int(0.25 * h))
                x1 = max(0, x1 - int(0.2 * w))
                x2 = min(img.shape[1], x2 + int(0.2 * w))

                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                label = test_antispoof(face_crop)
                if label == 1:
                    name = recognize2(face_crop, DB_DIR)
                    results.append({"status": "real", "name": name})
                else:
                    results.append({"status": "fake", "name": None})

            return Response({"results": results}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
DB_DIR = r"D:/Python project/Django/FaceAPI/Face/db"

class AddToDBAPIView(APIView):
    def post(self, request):
        serializer = DBImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            name = serializer.validated_data['name']

            np_img = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(img_rgb, model='hog')
            if not boxes:
                return Response({"error": "No face found in image"}, status=status.HTTP_400_BAD_REQUEST)

            top, right, bottom, left = max(boxes, key=lambda b: (b[2]-b[0]) * (b[1]-b[3]))
            encodings = face_recognition.face_encodings(img_rgb, known_face_locations=[(top, right, bottom, left)])

            if not encodings:
                return Response({"error": "Failed to generate encoding"}, status=status.HTTP_400_BAD_REQUEST)

            embedding = encodings[0]
            pkl_path = os.path.join(DB_DIR, f"{name}.pkl")

            with open(pkl_path, 'wb') as f:
                pickle.dump(embedding, f)

            return Response({"message": f"✅ Saved embedding as {name}.pkl"}, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)







# 🧠 FaceAPI – Face Recognition & Anti-Spoofing with Django

A robust Django REST API for real-time face authentication, powered by InsightFace, Silent-Face Anti-Spoofing, and face_recognition. Supports image uploads, spoof detection, and identity matching using precomputed embeddings.

---

## 🚀 Features

- Upload an image and detect faces
- Anti-spoofing check using Silent-Face models
- Face recognition using precomputed `.pkl` embeddings
- Add new faces to the database via API
- RESTful endpoints for easy integration

---

## 📦 Requirements

- Python 3.8+
- Django
- dlib
- face_recognition
- insightface
- OpenCV
- numpy
- torch
- onnxruntime

Install dependencies:

```bash
pip install -r requirements.txt
import os
import pickle

import face_recognition
import cv2
import numpy as np

def recognize(img, db_path):
    if img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    embeddings_unknown = face_recognition.face_encodings(img_rgb)
    if not embeddings_unknown:
        return 'no_persons_found'

    embeddings_unknown = embeddings_unknown[0]
    db_files = sorted(os.listdir(db_path))

    for file_name in db_files:
        file_path = os.path.join(db_path, file_name)
        with open(file_path, 'rb') as f:
            known_embedding = pickle.load(f)

        match = face_recognition.compare_faces([known_embedding], embeddings_unknown)[0]
        if match:
            return os.path.splitext(file_name)[0]

    return 'unknown_person'


def recognize2(img, db_path, tolerance=0.60, detector_model="hog"):
    """
    img: BGR (OpenCV) or RGB image of ONE face (crop is okay).
    db_path: folder of .pkl files (each containing one face embedding).
    tolerance: smaller => fewer false positives.
    detector_model: 'hog' (CPU, fast) or 'cnn' (GPU/slow but better).
    """

    # --- 1) Prepare unknown image ---
    if img.ndim == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    boxes_unknown = face_recognition.face_locations(img_rgb, model=detector_model)
    if not boxes_unknown:
        return "no_persons_found"

    top, right, bottom, left = max(boxes_unknown, key=lambda b: (b[2]-b[0]) * (b[1]-b[3]))
    unknown_encoding = face_recognition.face_encodings(
        img_rgb, known_face_locations=[(top, right, bottom, left)], num_jitters=1
    )
    if not unknown_encoding:
        return "no_persons_found"
    unknown_encoding = unknown_encoding[0]

    # --- 2) Load known encodings from .pkl files ---
    known_encodings = []
    known_names = []

    for fname in sorted(os.listdir(db_path)):
        if not fname.lower().endswith(".pkl"):
            continue

        path_ = os.path.join(db_path, fname)
        with open(path_, 'rb') as f:
            embedding = pickle.load(f)

        known_encodings.append(embedding)
        known_names.append(os.path.splitext(fname)[0])  # Strip .pkl

    if not known_encodings:
        return "db_empty_or_no_embeddings"

    # --- 3) Compare and return best match ---
    distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance <= tolerance:
        return known_names[best_idx]
    else:
        return "unknown_person"


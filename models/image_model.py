import numpy as np
import cv2
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import layers
from keras.saving import custom_object_scope  # This is crucial for TF >= 2.11
from config import COMMON_EMOTIONS
from utils.inference_utils import unify_probs


def load_image_model(json_path, weights_path):
    with open(json_path, "r") as f:
        model_json = f.read()

    with custom_object_scope({'Sequential': Sequential, 'Conv2D': layers.Conv2D,
                              'MaxPooling2D': layers.MaxPooling2D, 'Dropout': layers.Dropout,
                              'Flatten': layers.Flatten, 'Dense': layers.Dense}):
        model = model_from_json(model_json)

    model.load_weights(weights_path)
    return model


def extract_faces(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in face_rects:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face.astype("float32") / 255.0
                faces.append(np.expand_dims(face, axis=-1))
        count += 1
    cap.release()
    return np.array(faces)


def get_image_probs(video_path, model):
    faces = extract_faces(video_path)
    if len(faces) == 0:
        return {label: 0.0 for label in COMMON_EMOTIONS}
    preds = model.predict(faces, verbose=0)
    avg_probs = np.mean(preds, axis=0)
    labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return unify_probs(labels, avg_probs)

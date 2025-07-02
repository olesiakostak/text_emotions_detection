from vectorizer.SpacyVectorizer import SpacyVectorizer  # <-- This is key
import joblib
from config import TEXT_MODEL_PATH
from utils.inference_utils import unify_probs

def load_text_model():
    return joblib.load(TEXT_MODEL_PATH)

def get_text_probs(text, model):
    raw_probs = model.predict_proba([text])[0]
    raw_labels = list(model.classes_)
    return unify_probs(raw_labels, raw_probs)
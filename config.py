TEXT_MODEL_PATH = "saved_models/text_emotion.pkl"
AUDIO_MODEL_PATH = "saved_models/emotion_audio_recognition_model.keras"
IMAGE_MODEL_JSON = "saved_models/emotiondetector.json"
IMAGE_MODEL_WEIGHTS = "saved_models/emotiondetector.h5"
VIDEO_PATH = "data/your_video.mp4"
AUDIO_PATH = "data/temp_audio.wav"

TEXT_WEIGHT = 0.3
AUDIO_WEIGHT = 0.3
IMAGE_WEIGHT = 0.4

COMMON_EMOTIONS = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'happy']
LABEL_MAP = {
    "anger": "angry", "angry": "angry",
    "sadness": "sad", "sad": "sad",
    "joy": "happy", "happy": "happy", "ps": "happy",
    "shame": "sad", "surprise": "neutral",
    "disgust": "disgust", "fear": "fear", "neutral": "neutral"
}
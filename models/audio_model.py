import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip
from config import AUDIO_PATH, COMMON_EMOTIONS
from utils.inference_utils import unify_probs

def get_audio_probs(video_path, audio_model):
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(AUDIO_PATH, codec='pcm_s16le')
        clip.close()

        audio, sr = librosa.load(AUDIO_PATH, sr=22050, duration=300, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0)
        features = np.expand_dims(features, axis=0)

        raw_probs = audio_model.predict(features)[0]
        raw_labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps']
        return unify_probs(raw_labels, raw_probs)
    except Exception as e:
        print("Audio processing error:", e)
        return {label: 0.0 for label in COMMON_EMOTIONS}
    finally:
        if os.path.exists(AUDIO_PATH):
            os.remove(AUDIO_PATH)

from vectorizer.SpacyVectorizer import SpacyVectorizer  # Register class for unpickling

from config import (
    VIDEO_PATH, TEXT_WEIGHT, AUDIO_WEIGHT, IMAGE_WEIGHT,
    TEXT_MODEL_PATH, AUDIO_MODEL_PATH, IMAGE_MODEL_JSON, IMAGE_MODEL_WEIGHTS
)

from models.text_model import load_text_model, get_text_probs
from models.audio_model import get_audio_probs
from models.image_model import load_image_model, get_image_probs
from utils.preprocess import extract_audio, clean_text
from utils.inference_utils import merge_probs
import whisper


def transcribe_audio(audio_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


def main():
    print(f"\n🎬 Processing video: {VIDEO_PATH}\n")

    # Load models
    text_model = load_text_model()
    image_model = load_image_model(IMAGE_MODEL_JSON, IMAGE_MODEL_WEIGHTS)
    audio_model = None
    try:
        from tensorflow.keras.models import load_model
        audio_model = load_model(AUDIO_MODEL_PATH)
    except Exception as e:
        print("Failed to load audio model:", e)

    # Step 1: Transcribe text
    extract_audio(VIDEO_PATH, "data/temp_audio.wav")
    transcribed_text = transcribe_audio("data/temp_audio.wav")
    cleaned_text = clean_text(transcribed_text)
    print("📝 Transcribed text:", transcribed_text)

    # Step 2: Get probabilities
    print("\n🔍 Analyzing emotions...")

    text_probs = get_text_probs(cleaned_text, text_model)
    audio_probs = get_audio_probs(VIDEO_PATH, audio_model)
    image_probs = get_image_probs(VIDEO_PATH, image_model)

    emotions_emoji_dict = {
        "angry": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗",
        "neutral": "😐", "sad": "😔"
    }

    def print_model_probs(title, probs):
        print(f"\n🧠 {title} Emotion Probabilities:")
        for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            emoji = emotions_emoji_dict.get(emotion, "")
            print(f"{emotion.upper():<10} {emoji:<3} -> {prob*100:.2f}%")

    print_model_probs("Text", text_probs)
    print_model_probs("Audio", audio_probs)
    print_model_probs("Image", image_probs)

    # Step 3: Merge results
    combined_probs = merge_probs(text_probs, audio_probs, image_probs,
                                 TEXT_WEIGHT, AUDIO_WEIGHT, IMAGE_WEIGHT)
    sorted_combined = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)

    # Step 4: Display results
    print("\n🎯 Combined Emotion Probabilities:\n")
    for emotion, prob in sorted_combined:
        emoji = emotions_emoji_dict.get(emotion, "")
        print(f"{emotion.upper():<10} {emoji:<3} -> {prob*100:.2f}%")

    final_emotion = sorted_combined[0][0]
    print(f"\n🏆 Final Predicted Emotion: {final_emotion.upper()} {emotions_emoji_dict.get(final_emotion, '')}\n")


if __name__ == "__main__":
    main()

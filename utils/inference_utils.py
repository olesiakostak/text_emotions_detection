from config import COMMON_EMOTIONS, LABEL_MAP

def unify_probs(labels, probs) -> dict:
    mapped = {emotion: 0.0 for emotion in COMMON_EMOTIONS}
    for label, prob in zip(labels, probs):
        mapped_label = LABEL_MAP.get(label.lower())
        if mapped_label in mapped:
            mapped[mapped_label] += prob
    return mapped

def merge_probs(text_probs: dict, audio_probs: dict, image_probs: dict,
                text_weight: float, audio_weight: float, image_weight: float) -> dict:
    merged = {}
    for emotion in COMMON_EMOTIONS:
        merged[emotion] = (
            text_probs.get(emotion, 0) * text_weight +
            audio_probs.get(emotion, 0) * audio_weight +
            image_probs.get(emotion, 0) * image_weight
        )
    return merged

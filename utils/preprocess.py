import ffmpeg
import neattext.functions as nfx

def extract_audio(video_path: str, audio_path: str):
    ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True)

def clean_text(text: str) -> str:
    text = nfx.remove_userhandles(text)
    text = nfx.remove_stopwords(text)
    return text
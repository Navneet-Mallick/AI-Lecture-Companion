import whisper


MODEL_NAME = "base"


def load_whisper_model():
    """Load and return the Whisper speech recognition model."""
    return whisper.load_model(MODEL_NAME)


def transcribe_audio(model, audio_path: str) -> str:
    """Transcribe an audio or video file using Whisper."""
    result = model.transcribe(audio_path)
    return result["text"].strip()
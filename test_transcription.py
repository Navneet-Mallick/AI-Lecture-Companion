from utils.speech import load_whisper_model, transcribe_audio


AUDIO_FILE = "audio/sample.mp4"


print("Loading Whisper model...")
model = load_whisper_model()

print("Whisper loaded successfully!")
print("Transcribing lecture...")

transcript = transcribe_audio(model, AUDIO_FILE)

print("\n--- TRANSCRIPT ---\n")
print(transcript)
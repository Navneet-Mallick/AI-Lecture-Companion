from utils.speech import load_whisper_model, transcribe_audio
from utils.llm import generate_study_material


AUDIO_FILE = "audio/sample.mp4"


print("========================================")
print("       AI LECTURE COMPANION")
print("========================================")

# ----------------------------------------
# STEP 1: Speech Recognition
# ----------------------------------------

print("\n[1/2] Loading Whisper...")

whisper_model = load_whisper_model()

print("Whisper loaded.")
print("Transcribing lecture...")

transcript = transcribe_audio(
    whisper_model,
    AUDIO_FILE
)

print("\n--- TRANSCRIPT ---")
print(transcript)


# ----------------------------------------
# STEP 2: Generative AI
# ----------------------------------------

print("\n[2/2] Sending transcript to Gemini...")
print("Generating study material...")

study_material = generate_study_material(transcript)


# ----------------------------------------
# Display Results
# ----------------------------------------

print("\n--- SUMMARY ---")
print(study_material["summary"])


print("\n--- KEY CONCEPTS ---")

for concept in study_material["key_concepts"]:
    print(f"\n{concept['concept']}")
    print(concept["explanation"])


print("\n--- QUIZ ---")

for index, question in enumerate(
    study_material["quiz"],
    start=1
):
    print(f"\n{index}. {question['question']}")

    for option in question["options"]:
        print(f"   - {option}")

    print(f"Correct Answer: {question['correct_answer']}")
    print(f"Explanation: {question['explanation']}")


print("\n========================================")
print("           PIPELINE COMPLETE")
print("========================================")
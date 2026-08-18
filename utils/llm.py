from utils.gemini import answer_question, generate_all_study_material


# Whisper handles speech recognition.
# Gemini handles lecture summary, concept extraction, quiz generation, and lecture Q&A.


def generate_study_material(transcript: str) -> dict:
    """Generate study material using a single Gemini API call for speed."""
    return generate_all_study_material(transcript)


def ask_lecture(transcript: str, question: str) -> str:
    """Answer a lecture question using Gemini grounded in the transcript."""
    return answer_question(transcript, question)

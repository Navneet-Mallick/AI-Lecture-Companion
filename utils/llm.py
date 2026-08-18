from utils.gemini import answer_question, extract_key_concepts, generate_quiz, generate_summary


# Whisper handles speech recognition.
# Gemini handles lecture summary, concept extraction, quiz generation, and lecture Q&A.


def generate_study_material(transcript: str) -> dict:
    """Generate study material in the exact structure expected by the Streamlit app using Gemini."""
    summary = generate_summary(transcript)
    key_concepts = extract_key_concepts(transcript, summary)
    quiz = generate_quiz(summary, transcript, key_concepts)

    return {
        "summary": summary,
        "key_concepts": key_concepts,
        "quiz": quiz,
    }


def ask_lecture(transcript: str, question: str) -> str:
    """Answer a lecture question using Gemini grounded in the transcript."""
    return answer_question(transcript, question)


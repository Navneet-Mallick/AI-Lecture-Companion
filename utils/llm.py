from utils.gemini import (
    answer_question,
    generate_all_study_material,
    generate_summary,
    extract_key_concepts,
    generate_quiz,
)


def generate_study_material(transcript: str) -> dict:
    """Generate all study material in a single Gemini API call."""
    return generate_all_study_material(transcript)


def generate_summary_only(transcript: str) -> str:
    """Generate just the summary."""
    return generate_summary(transcript)


def generate_concepts_only(transcript: str, summary: str) -> list:
    """Generate just the key concepts."""
    return extract_key_concepts(transcript, summary)


def generate_quiz_only(summary: str, transcript: str, key_concepts=None) -> list:
    """Generate just the quiz."""
    return generate_quiz(summary, transcript, key_concepts)


def ask_lecture(transcript: str, question: str) -> str:
    """Answer a lecture question using Gemini grounded in the transcript."""
    return answer_question(transcript, question)

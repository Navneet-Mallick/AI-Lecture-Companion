import json
import os
import re

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - dependency may be missing until installed
    genai = None


def _get_api_key() -> str:
    load_dotenv()
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _get_model():
    if genai is None:
        raise RuntimeError("google-generativeai is not installed. Please install it with pip install google-generativeai")

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

    genai.configure(api_key=api_key)

    model_names = [
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    last_error = None
    for model_name in model_names:
        try:
            return genai.GenerativeModel(model_name)
        except Exception as exc:  # pragma: no cover - tried multiple names for compatibility
            last_error = exc

    raise RuntimeError(
        "No supported Gemini model was available for this account. Tried: "
        + ", ".join(model_names)
    ) from last_error


def _extract_text(response) -> str:
    if response is None:
        return ""

    text = getattr(response, "text", "")
    if text:
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        parts = getattr(candidate, "content", None)
        if parts is None:
            continue
        for part in getattr(parts, "parts", []) or []:
            if getattr(part, "text", None):
                return part.text.strip()

    return str(response).strip()


def _extract_json_array(raw_text: str):
    text = (raw_text or "").strip()
    if not text:
        return []

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []


def generate_summary(transcript: str) -> str:
    transcript = (transcript or "").strip()
    if not transcript:
        return "No transcript content was provided."

    model = _get_model()
    prompt = (
        "You are generating study notes from a lecture transcript. "
        "Write a concise but complete summary in clear English. "
        "Keep it focused on the lecture's main ideas, examples, and key takeaways. "
        "Do not invent facts.\n\nTranscript:\n"
        f"{transcript}"
    )
    response = model.generate_content(prompt)
    summary = _extract_text(response)
    return summary or transcript[:800]


def extract_key_concepts(transcript: str, summary: str = "", max_concepts: int = 5):
    transcript = (transcript or "").strip()
    summary = (summary or transcript or "").strip()
    if not summary:
        return [{"concept": "No content", "explanation": "No lecture content was provided."}]

    model = _get_model()
    prompt = (
        "Extract up to 5 key concepts from this lecture summary. "
        "Return valid JSON as an array of objects with keys: concept and explanation. "
        "Each concept should be a short title and each explanation should be a brief, accurate statement.\n\n"
        f"Summary:\n{summary}"
    )
    response = model.generate_content(prompt)
    raw_text = _extract_text(response)
    parsed = _extract_json_array(raw_text)

    if parsed and isinstance(parsed, list):
        cleaned = []
        for item in parsed[:max_concepts]:
            if not isinstance(item, dict):
                continue
            concept = str(item.get("concept") or "Key Idea").strip()
            explanation = str(item.get("explanation") or item.get("description") or summary).strip()
            if concept:
                cleaned.append({"concept": concept, "explanation": explanation})
        if cleaned:
            return cleaned

    fallback = [
        {"concept": "Primary topic", "explanation": summary[:250]},
        {"concept": "Core idea", "explanation": "The lecture explains the main concept with supporting examples and definitions."},
    ]
    return fallback[:max_concepts]


def generate_quiz(summary: str, transcript: str = "", key_concepts=None):
    source = summary or transcript or ""
    if not source:
        return [{
            "question": "No lecture content was provided.",
            "options": ["No content available", "No content available", "No content available", "No content available"],
            "correct_answer": "No content available",
            "explanation": "No lecture transcript was available to generate quiz questions."
        }]

    model = _get_model()
    key_context = ""
    if key_concepts:
        key_context = "Key concepts:\n" + "\n".join(f"- {item.get('concept', 'Key idea')}: {item.get('explanation', '')}" for item in key_concepts[:5]) + "\n\n"

    prompt = (
        "Generate exactly 5 multiple-choice questions based only on the lecture content. "
        "Return valid JSON only, as an array of objects with keys: question, options, correct_answer, explanation. "
        "Each question must have exactly 4 options. The correct_answer must match one option exactly. "
        "Do not include any extra text outside the JSON.\n\n"
        f"{key_context}"
        f"Lecture summary:\n{summary}\n\nLecture transcript:\n{transcript[:4000]}"
    )
    response = model.generate_content(prompt)
    raw_text = _extract_text(response)
    parsed = _extract_json_array(raw_text)

    if parsed and isinstance(parsed, list):
        cleaned = []
        for item in parsed[:5]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            options = item.get("options") or []
            if not question or not isinstance(options, list) or len(options) != 4:
                continue
            options = [str(opt).strip() for opt in options[:4]]
            correct_answer = str(item.get("correct_answer") or item.get("answer") or "").strip()
            if correct_answer not in options:
                correct_answer = options[0]
            explanation = str(item.get("explanation") or "This answer is supported by the lecture content.").strip()
            cleaned.append({
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": explanation,
            })
        if cleaned:
            return cleaned

    return [{
        "question": "Which statement best matches the lecture?",
        "options": [
            summary[:180] if summary else "The lecture explains the main concepts.",
            "A topic unrelated to the lecture.",
            "A false claim not supported by the lecture.",
            "A random statement with no instructional value."
        ],
        "correct_answer": summary[:180] if summary else "The lecture explains the main concepts.",
        "explanation": "This is the closest lecture-grounded statement derived from the provided material."
    }]


def answer_question(transcript: str, question: str) -> str:
    transcript = (transcript or "").strip()
    question = (question or "").strip()

    if not transcript:
        return "No transcript is available for question answering."
    if not question:
        return "Please enter a question before asking the lecture."

    model = _get_model()
    prompt = (
        "Answer the user's question using only the lecture transcript as the source. "
        "If the transcript does not contain enough information, say that the information is not covered. "
        "Keep the response concise and grounded in the lecture.\n\n"
        f"Question: {question}\n\nTranscript:\n{transcript[:12000]}"
    )
    response = model.generate_content(prompt)
    answer = _extract_text(response)
    if answer:
        return answer
    return "This information is not covered in the lecture."

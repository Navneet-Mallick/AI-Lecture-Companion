import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - dependency may be missing until installed
    genai = None


def _get_api_key() -> str:
    load_dotenv()
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


@lru_cache(maxsize=1)
def _get_model():
    """Load and cache the Gemini model instance."""
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
        except Exception as exc:
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


def _extract_json(raw_text: str):
    """Extract a JSON object or array from model output."""
    text = (raw_text or "").strip()
    if not text:
        return None

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # Try parsing as-is first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    # Try to find a JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return None


def generate_all_study_material(transcript: str) -> dict:
    """Generate summary, key concepts, and quiz in a SINGLE Gemini API call for speed."""
    transcript = (transcript or "").strip()
    if not transcript:
        return {
            "summary": "No transcript content was provided.",
            "key_concepts": [{"concept": "No content", "explanation": "No lecture content was provided."}],
            "quiz": [],
        }

    model = _get_model()

    # Limit transcript size to keep the request fast
    trimmed_transcript = transcript[:8000]

    prompt = (
        "You are an AI study assistant. Based on the lecture transcript below, generate ALL of the following in a single JSON response:\n\n"
        "1. **summary**: A concise but complete summary of the lecture (2-4 paragraphs).\n"
        "2. **key_concepts**: An array of up to 5 objects, each with \"concept\" (short title) and \"explanation\" (brief description).\n"
        "3. **quiz**: An array of exactly 5 multiple-choice questions. Each object must have:\n"
        "   - \"question\": the question text\n"
        "   - \"options\": array of exactly 4 answer choices\n"
        "   - \"correct_answer\": must exactly match one of the options\n"
        "   - \"explanation\": why the answer is correct\n\n"
        "Return ONLY valid JSON with keys: summary, key_concepts, quiz. No extra text.\n\n"
        f"Transcript:\n{trimmed_transcript}"
    )

    response = model.generate_content(prompt)
    raw_text = _extract_text(response)
    parsed = _extract_json(raw_text)

    result = {
        "summary": "",
        "key_concepts": [],
        "quiz": [],
    }

    if isinstance(parsed, dict):
        # Extract summary
        result["summary"] = str(parsed.get("summary") or "").strip()

        # Extract key concepts
        concepts_raw = parsed.get("key_concepts") or []
        if isinstance(concepts_raw, list):
            for item in concepts_raw[:5]:
                if isinstance(item, dict):
                    concept = str(item.get("concept") or "Key Idea").strip()
                    explanation = str(item.get("explanation") or item.get("description") or "").strip()
                    if concept:
                        result["key_concepts"].append({"concept": concept, "explanation": explanation})

        # Extract quiz
        quiz_raw = parsed.get("quiz") or []
        if isinstance(quiz_raw, list):
            for item in quiz_raw[:5]:
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
                result["quiz"].append({
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer,
                    "explanation": explanation,
                })

    # Fallbacks if parsing failed
    if not result["summary"]:
        result["summary"] = transcript[:800]
    if not result["key_concepts"]:
        result["key_concepts"] = [
            {"concept": "Primary topic", "explanation": result["summary"][:250]},
            {"concept": "Core idea", "explanation": "The lecture explains the main concept with supporting examples."},
        ]
    if not result["quiz"]:
        result["quiz"] = [{
            "question": "Which statement best matches the lecture?",
            "options": [
                result["summary"][:180],
                "A topic unrelated to the lecture.",
                "A false claim not supported by the lecture.",
                "A random statement with no instructional value."
            ],
            "correct_answer": result["summary"][:180],
            "explanation": "This is the closest lecture-grounded statement from the provided material."
        }]

    return result


def generate_summary(transcript: str) -> str:
    """Generate a summary (standalone, used if needed individually)."""
    transcript = (transcript or "").strip()
    if not transcript:
        return "No transcript content was provided."

    model = _get_model()
    prompt = (
        "Write a concise but complete summary of this lecture transcript. "
        "Focus on main ideas, examples, and key takeaways. Do not invent facts.\n\n"
        f"Transcript:\n{transcript[:8000]}"
    )
    response = model.generate_content(prompt)
    summary = _extract_text(response)
    return summary or transcript[:800]


def extract_key_concepts(transcript: str, summary: str = "", max_concepts: int = 5):
    """Extract key concepts (standalone, used if needed individually)."""
    summary = (summary or transcript or "").strip()
    if not summary:
        return [{"concept": "No content", "explanation": "No lecture content was provided."}]

    model = _get_model()
    prompt = (
        "Extract up to 5 key concepts from this lecture summary. "
        "Return valid JSON as an array of objects with keys: concept and explanation.\n\n"
        f"Summary:\n{summary}"
    )
    response = model.generate_content(prompt)
    raw_text = _extract_text(response)
    parsed = _extract_json(raw_text)

    if isinstance(parsed, list):
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

    return [
        {"concept": "Primary topic", "explanation": summary[:250]},
        {"concept": "Core idea", "explanation": "The lecture explains the main concept with supporting examples."},
    ]


def generate_quiz(summary: str, transcript: str = "", key_concepts=None):
    """Generate quiz (standalone, used if needed individually)."""
    source = summary or transcript or ""
    if not source:
        return []

    model = _get_model()
    prompt = (
        "Generate exactly 5 multiple-choice questions based on this lecture. "
        "Return valid JSON only as an array of objects with keys: question, options, correct_answer, explanation. "
        "Each question must have exactly 4 options. correct_answer must match one option exactly.\n\n"
        f"Lecture summary:\n{summary}\n\nTranscript:\n{(transcript or '')[:4000]}"
    )
    response = model.generate_content(prompt)
    raw_text = _extract_text(response)
    parsed = _extract_json(raw_text)

    if isinstance(parsed, list):
        cleaned = []
        for item in parsed[:5]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            options = item.get("options") or []
            if not question or not isinstance(options, list) or len(options) != 4:
                continue
            options = [str(opt).strip() for opt in options[:4]]
            correct_answer = str(item.get("correct_answer") or "").strip()
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
            source[:180],
            "A topic unrelated to the lecture.",
            "A false claim not supported by the lecture.",
            "A random statement with no instructional value."
        ],
        "correct_answer": source[:180],
        "explanation": "This is the closest lecture-grounded statement from the provided material."
    }]


def answer_question(transcript: str, question: str) -> str:
    """Answer a question using the lecture transcript."""
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
        f"Question: {question}\n\nTranscript:\n{transcript[:8000]}"
    )
    response = model.generate_content(prompt)
    answer = _extract_text(response)
    if answer:
        return answer
    return "This information is not covered in the lecture."

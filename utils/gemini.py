import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from google import genai


def _get_api_key() -> str:
    load_dotenv()
    return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()


@lru_cache(maxsize=1)
def _get_client():
    """Create and cache the Gemini client."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


MODEL_NAME = "gemini-2.5-flash"


def _generate(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return (response.text or "").strip()


def _extract_json(raw_text: str):
    """Extract a JSON object or array from model output."""
    text = (raw_text or "").strip()
    if not text:
        return None

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # Try parsing as-is
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
    """Generate summary, key concepts, and quiz in a single Gemini call."""
    transcript = (transcript or "").strip()
    if not transcript:
        return {
            "summary": "No transcript content was provided.",
            "key_concepts": [{"concept": "No content", "explanation": "No lecture content was provided."}],
            "quiz": [],
        }

    trimmed = transcript[:8000]
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
        f"Transcript:\n{trimmed}"
    )

    raw = _generate(prompt)
    parsed = _extract_json(raw)

    result = {"summary": "", "key_concepts": [], "quiz": []}

    if isinstance(parsed, dict):
        result["summary"] = str(parsed.get("summary") or "").strip()

        for item in (parsed.get("key_concepts") or [])[:5]:
            if isinstance(item, dict):
                concept = str(item.get("concept") or "Key Idea").strip()
                explanation = str(item.get("explanation") or item.get("description") or "").strip()
                if concept:
                    result["key_concepts"].append({"concept": concept, "explanation": explanation})

        for item in (parsed.get("quiz") or [])[:5]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            options = item.get("options") or []
            if not question or not isinstance(options, list) or len(options) != 4:
                continue
            options = [str(o).strip() for o in options[:4]]
            correct = str(item.get("correct_answer") or item.get("answer") or "").strip()
            if correct not in options:
                correct = options[0]
            explanation = str(item.get("explanation") or "Supported by the lecture.").strip()
            result["quiz"].append({
                "question": question,
                "options": options,
                "correct_answer": correct,
                "explanation": explanation,
            })

    # Fallbacks
    if not result["summary"]:
        result["summary"] = transcript[:800]
    if not result["key_concepts"]:
        result["key_concepts"] = [{"concept": "Main topic", "explanation": result["summary"][:250]}]
    if not result["quiz"]:
        result["quiz"] = [{
            "question": "Which statement best matches the lecture?",
            "options": [result["summary"][:150], "Unrelated topic", "False claim", "Random statement"],
            "correct_answer": result["summary"][:150],
            "explanation": "Derived from the lecture content."
        }]

    return result


def generate_summary(transcript: str) -> str:
    """Generate a summary."""
    transcript = (transcript or "").strip()
    if not transcript:
        return "No transcript content was provided."

    prompt = (
        "Write a concise but complete summary of this lecture transcript. "
        "Focus on main ideas, examples, and key takeaways. Do not invent facts.\n\n"
        f"Transcript:\n{transcript[:8000]}"
    )
    result = _generate(prompt)
    return result or transcript[:800]


def extract_key_concepts(transcript: str, summary: str = "", max_concepts: int = 5):
    """Extract key concepts."""
    summary = (summary or transcript or "").strip()
    if not summary:
        return [{"concept": "No content", "explanation": "No lecture content was provided."}]

    prompt = (
        "Extract up to 5 key concepts from this lecture summary. "
        "Return valid JSON as an array of objects with keys: concept and explanation.\n\n"
        f"Summary:\n{summary}"
    )
    raw = _generate(prompt)
    parsed = _extract_json(raw)

    if isinstance(parsed, list):
        cleaned = []
        for item in parsed[:max_concepts]:
            if isinstance(item, dict):
                concept = str(item.get("concept") or "Key Idea").strip()
                explanation = str(item.get("explanation") or "").strip()
                if concept:
                    cleaned.append({"concept": concept, "explanation": explanation})
        if cleaned:
            return cleaned

    return [{"concept": "Main topic", "explanation": summary[:250]}]


def generate_quiz(summary: str, transcript: str = "", key_concepts=None):
    """Generate quiz questions."""
    source = summary or transcript or ""
    if not source:
        return []

    prompt = (
        "Generate exactly 5 multiple-choice questions based on this lecture. "
        "Return valid JSON only as an array of objects with keys: question, options, correct_answer, explanation. "
        "Each question must have exactly 4 options. correct_answer must match one option exactly.\n\n"
        f"Lecture summary:\n{summary}\n\nTranscript:\n{(transcript or '')[:4000]}"
    )
    raw = _generate(prompt)
    parsed = _extract_json(raw)

    if isinstance(parsed, list):
        cleaned = []
        for item in parsed[:5]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            options = item.get("options") or []
            if not question or not isinstance(options, list) or len(options) != 4:
                continue
            options = [str(o).strip() for o in options[:4]]
            correct = str(item.get("correct_answer") or "").strip()
            if correct not in options:
                correct = options[0]
            explanation = str(item.get("explanation") or "Supported by the lecture.").strip()
            cleaned.append({
                "question": question, "options": options,
                "correct_answer": correct, "explanation": explanation,
            })
        if cleaned:
            return cleaned

    return [{
        "question": "Which statement best matches the lecture?",
        "options": [source[:150], "Unrelated topic", "False claim", "Random statement"],
        "correct_answer": source[:150],
        "explanation": "Derived from the lecture content."
    }]


def answer_question(transcript: str, question: str) -> str:
    """Answer a question using the lecture transcript."""
    transcript = (transcript or "").strip()
    question = (question or "").strip()

    if not transcript:
        return "No transcript is available for question answering."
    if not question:
        return "Please enter a question."

    prompt = (
        "Answer the user's question using only the lecture transcript as the source. "
        "If the transcript does not contain enough information, say so. "
        "Keep the response concise.\n\n"
        f"Question: {question}\n\nTranscript:\n{transcript[:8000]}"
    )
    result = _generate(prompt)
    return result or "This information is not covered in the lecture."

import re
from functools import lru_cache

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from transformers import pipeline


QUIZ_MODEL_NAME = "google/flan-t5-base"


@lru_cache(maxsize=1)
def load_quiz_generator():
    """Load the pretrained FLAN-T5-Base text-generation model.
    
    Auto-detects GPU (device 0) if available, falls back to CPU (-1).
    """
    device = 0 if (HAS_TORCH and torch.cuda.is_available()) else -1
    
    try:
        return pipeline(
            "text-generation",
            model=QUIZ_MODEL_NAME,
            device=device,
        )
    except Exception:
        return None


def _generate_fallback_quiz(summary: str, key_concepts):
    if not summary:
        summary = "This lecture introduces the main ideas and important learning objectives."

    concept_items = key_concepts or [{"concept": "Main Topic", "explanation": summary}]
    concept_titles = [item["concept"] for item in concept_items]
    concept_explanations = [item["explanation"] for item in concept_items]

    questions = []
    
    # Q1: Multiple concept choices
    if len(concept_titles) >= 2:
        q1_options = [
            concept_titles[0],
            concept_titles[1],
            "Random unrelated topic",
            "Not mentioned in the lecture"
        ]
    else:
        q1_options = [
            concept_titles[0] if concept_titles else "Artificial Intelligence",
            "Unrelated concept",
            "Not covered",
            "A different field"
        ]
    questions.append({
        "question": f"What is one of the main topics covered in this lecture?",
        "options": q1_options,
        "correct_answer": q1_options[0],
        "explanation": "This concept was discussed in the lecture."
    })
    
    # Q2: Summary match (using full summary, not truncated)
    # Split at sentence boundary to avoid cutting mid-word
    summary_for_q2 = summary
    if len(summary) > 200:
        # Find last period before 200 chars
        truncated = summary[:200]
        last_period = truncated.rfind('.')
        if last_period > 100:  # Only truncate if there's a decent chunk
            summary_for_q2 = truncated[:last_period+1]
        else:
            summary_for_q2 = truncated
    
    q2_options = [
        summary_for_q2,
        "A completely unrelated field",
        "The opposite of what was taught",
        "A future prediction, not current content"
    ]
    questions.append({
        "question": "Which best describes the lecture content?",
        "options": q2_options,
        "correct_answer": q2_options[0],
        "explanation": "This summary matches the lecture material."
    })
    
    # Q3: Purpose
    questions.append({
        "question": "What is the main purpose of this lecture?",
        "options": [
            "To explain key concepts and their definitions",
            "To avoid covering important topics",
            "To confuse students",
            "To discuss unrelated subjects"
        ],
        "correct_answer": "To explain key concepts and their definitions",
        "explanation": "The lecture aims to teach and explain important concepts."
    })
    
    # Q4: Specific concept (if available)
    if len(concept_explanations) > 0:
        q4_options = [
            concept_explanations[0],
            "Something completely different",
            "An outdated definition",
            "Not relevant to this lecture"
        ]
    else:
        q4_options = [
            "The lecture explains foundational concepts",
            "Random unrelated fact",
            "False information",
            "Opposite of the lecture"
        ]
    questions.append({
        "question": "What key idea was emphasized in the lecture?",
        "options": q4_options,
        "correct_answer": q4_options[0],
        "explanation": "This reflects the core content of the lecture."
    })
    
    # Q5: Application/Understanding
    questions.append({
        "question": "Based on the lecture, which statement is TRUE?",
        "options": [
            concept_titles[0] if concept_titles else "The lecture covers important topics",
            "The lecture was not about what was described",
            "Nothing important was discussed",
            "All concepts were unrelated"
        ],
        "correct_answer": concept_titles[0] if concept_titles else "The lecture covers important topics",
        "explanation": "This statement is supported by the lecture content."
    })

    return questions[:5]


def _parse_generated_quiz(text: str):
    """Try to parse a question list from model output; otherwise fall back safely."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    pattern = re.compile(r"(?:Q(?:uestion)?\s*[:\-]?\s*)(.+?)(?:\n|$)\s*(?:A\)|A\.|A\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:B\)|B\.|B\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:C\)|C\.|C\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:D\)|D\.|D\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:Correct|Answer)\s*[:\-]?\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)

    match = pattern.search(cleaned)
    if match:
        question = match.group(1).strip()
        options = [
            match.group(2).strip(),
            match.group(3).strip(),
            match.group(4).strip(),
            match.group(5).strip(),
        ]
        correct_answer = match.group(6).strip()
        explanation = "This answer is supported by the lecture content."
        return [{
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
        }]

    return []


def generate_quiz(summary: str, transcript: str = "", key_concepts=None):
    """Generate multiple-choice questions using a lightweight pretrained Hugging Face generation model."""
    if not summary:
        summary = transcript[:800] if transcript else "This lecture discusses the key concepts and important learning objectives."

    if key_concepts is None:
        key_concepts = []

    try:
        generator = load_quiz_generator()
        prompt = (
            "Generate 5 multiple-choice questions based only on the lecture summary below. "
            "Return each question with exactly four options labeled A, B, C, D and include the correct answer. "
            "Keep the answers grounded in the lecture.\n\nSummary:\n"
            f"{summary}"
        )
        result = generator(prompt, max_length=400, num_beams=4, do_sample=False)
        text = result[0].get("generated_text", "") if result else ""
        parsed = _parse_generated_quiz(text)
        if parsed:
            return parsed[:5]
    except Exception:
        pass

    return _generate_fallback_quiz(summary, key_concepts)

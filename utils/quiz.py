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
    summary = (summary or "").strip() or "This lecture introduces the main ideas and learning objectives."

    concept_items = key_concepts or []
    if not concept_items:
        concept_items = [{"concept": "Main Topic", "explanation": summary}]

    concept_titles = [str(item.get("concept", "")).strip() for item in concept_items if str(item.get("concept", "")).strip()]
    concept_explanations = [str(item.get("explanation", summary)).strip() for item in concept_items if str(item.get("explanation", "")).strip()]

    if not concept_titles:
        concept_titles = ["Main Topic"]
    if not concept_explanations:
        concept_explanations = [summary]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    main_claim = sentences[0] if sentences else summary

    questions = []

    # Question 1: anchored to a real concept from the lecture
    first_concept = concept_titles[0]
    first_explanation = concept_explanations[0]
    distractors = []
    for other in concept_titles[1:]:
        distractors.append(other)
    if len(distractors) < 3:
        distractors.extend([
            "A future prediction unrelated to the lecture",
            "An idea not mentioned in the lecture",
            "A completely different field"
        ])

    question_1_options = [first_explanation, distractors[0], distractors[1], distractors[2]]
    questions.append({
        "question": f"Which statement best describes {first_concept}?",
        "options": question_1_options,
        "correct_answer": first_explanation,
        "explanation": f"{first_concept} is discussed in the lecture as follows: {first_explanation}"
    })

    # Question 2: grounded in the actual summary claim
    support_options = [
        main_claim,
        "A topic that was not covered in the lecture",
        "A completely unrelated field",
        "A misleading statement contradicted by the lecture"
    ]
    questions.append({
        "question": "Which statement is supported by the lecture?",
        "options": support_options,
        "correct_answer": main_claim,
        "explanation": "This statement matches the lecture's actual content and main takeaway."
    })

    # Question 3: use a concrete second concept when available
    if len(concept_titles) >= 2:
        second_concept = concept_titles[1]
        second_explanation = concept_explanations[1] if len(concept_explanations) > 1 else concept_explanations[0]
        q3_options = [
            second_explanation,
            f"{second_concept} is completely unrelated to the lecture",
            "The lecture never discusses the topic",
            "This concept is the opposite of the lecture"
        ]
        questions.append({
            "question": f"Which option best matches the lecture's discussion of {second_concept}?",
            "options": q3_options,
            "correct_answer": second_explanation,
            "explanation": f"The lecture discusses {second_concept} in this way: {second_explanation}"
        })
    else:
        questions.append({
            "question": "What is the main purpose of this lecture?",
            "options": [
                "To explain the key concepts and ideas discussed",
                "To avoid discussing important ideas",
                "To make the topic confusing",
                "To cover unrelated subjects"
            ],
            "correct_answer": "To explain the key concepts and ideas discussed",
            "explanation": "The lecture is designed to teach the central concepts from the material."
        })

    # Question 4: directly references key concepts
    q4_options = [
        first_concept,
        "A random topic that was never introduced",
        "A future event not covered in the lecture",
        "A concept that contradicts the lecture"
    ]
    questions.append({
        "question": "Which option is a concept explicitly covered in the lecture?",
        "options": q4_options,
        "correct_answer": first_concept,
        "explanation": f"{first_concept} is one of the concepts directly discussed in the lecture material."
    })

    # Question 5: lecture understanding
    q5_options = [
        "The lecture explains important ideas using a concrete summary of the topic",
        "The lecture is entirely unrelated to the summary",
        "The lecture avoids discussing the main subject",
        "The lecture only contains random details"
    ]
    questions.append({
        "question": "Which statement is true about this lecture?",
        "options": q5_options,
        "correct_answer": "The lecture explains important ideas using a concrete summary of the topic",
        "explanation": "The lecture content is organized around a clear set of concepts and a coherent summary."
    })

    return questions[:5]


def _parse_generated_quiz(text: str):
    """Parse a model response if it contains a recognizable question list or a JSON object."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    # JSON-first parsing is more reliable for modern generation models.
    try:
        if cleaned.startswith("[") or cleaned.startswith("{"):
            parsed = __import__("json").loads(cleaned)
            if isinstance(parsed, list):
                questions = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    if "question" in item and "options" in item and len(item["options"]) >= 4:
                        options = [str(opt).strip() for opt in item["options"][:4]]
                        correct_answer = str(item.get("correct_answer") or item.get("answer") or "").strip()
                        if correct_answer and correct_answer in options:
                            questions.append({
                                "question": str(item["question"]).strip(),
                                "options": options,
                                "correct_answer": correct_answer,
                                "explanation": item.get("explanation") or "This answer is supported by the lecture content.",
                            })
                if questions:
                    return questions[:5]
    except Exception:
        pass

    pattern = re.compile(
        r"(?:Q(?:uestion)?\s*[:\-]?\s*)(.+?)(?:\n|$)\s*(?:A\)|A\.|A\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:B\)|B\.|B\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:C\)|C\.|C\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:D\)|D\.|D\s*[:\-])\s*(.+?)(?:\n|$)\s*(?:Correct|Answer)\s*[:\-]?\s*(.+?)(?:\n|$)",
        re.IGNORECASE | re.DOTALL,
    )

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

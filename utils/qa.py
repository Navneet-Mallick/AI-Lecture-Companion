from functools import lru_cache

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from transformers import pipeline


QA_MODEL_NAME = "distilbert-base-uncased-distilled-squad"


@lru_cache(maxsize=1)
def load_qa_model():
    """Load the pretrained DistilBERT QA model.
    
    Auto-detects GPU (device 0) if available, falls back to CPU (-1).
    """
    device = 0 if (HAS_TORCH and torch.cuda.is_available()) else -1
    
    try:
        return pipeline(
            "question-answering",
            model=QA_MODEL_NAME,
            device=device,
        )
    except Exception:
        return None


def _chunk_text(text: str, max_chars: int = 1200):
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        for sentence in paragraph.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 <= max_chars:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
            current = ""

    if not chunks:
        return [text[:max_chars]]

    return chunks[:10]


def answer_question(transcript: str, question: str) -> str:
    """Answer a question using the lecture transcript as the source context."""
    transcript_text = (transcript or "").strip()
    question_text = (question or "").strip()

    if not transcript_text:
        return "No transcript is available for question answering."
    if not question_text:
        return "Please enter a question before asking the lecture."

    try:
        qa_model = load_qa_model()
        if not qa_model:
            raise Exception("QA model failed to load")

        best_answer = None
        best_score = 0.0

        for chunk in _chunk_text(transcript_text):
            result = qa_model(question=question_text, context=chunk)
            answer = (result or {}).get("answer", "").strip()
            score = float((result or {}).get("score", 0.0) or 0.0)

            if answer and score > 0.01:
                best_answer = answer
                best_score = score
                if score > 0.3:
                    break

        if best_answer:
            return best_answer

    except Exception:
        pass

    # Strong transcript-grounded fallback: score sentences based on overlap with the question.
    question_lower = question_text.lower()
    question_tokens = [
        t for t in __import__("re").findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", question_lower)
        if t not in {
            "what", "when", "where", "which", "who", "why", "how", "does",
            "have", "from", "with", "your", "this", "that", "about", "used",
            "into", "over", "under", "after", "before", "through"
        }
    ]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript_text) if s.strip()]
    scored_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(1 for token in question_tokens if token in sent_lower)
        if score > 0:
            scored_sentences.append((score, sent.strip()))

    if scored_sentences:
        best_score_sentence = max(scored_sentences, key=lambda x: x[0])[1]
        return best_score_sentence

    # Final fallback: return a short, lecture-grounded response using the first meaningful sentences.
    stripped = [s.strip() for s in sentences[:3] if s.strip()]
    if stripped:
        return stripped[0]

    return "This information is not covered in the lecture. Try asking about the lecture's main concepts and examples."

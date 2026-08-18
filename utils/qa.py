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

            # Lowered threshold to 0.01 to accept weak matches
            if answer and score > 0.01:
                best_answer = answer
                best_score = score
                # Accept first decent match to avoid waiting for perfect score
                if score > 0.3:
                    break

        if best_answer:
            return best_answer

    except Exception:
        pass

    # Fallback: keyword extraction from transcript
    question_lower = question_text.lower()
    keywords = [w for w in question_lower.split() if len(w) > 3 and w not in ['what', 'when', 'where', 'which', 'does', 'have', 'from', 'with', 'your', 'this', 'that', 'about']]
    
    transcript_lower = transcript_text.lower()
    sentences = [s.strip() for s in transcript_text.split('.') if s.strip()]
    
    # Find sentences with matching keywords
    scored_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        matches = sum(1 for kw in keywords if kw in sent_lower)
        if matches > 0:
            scored_sentences.append((sent, matches))
    
    if scored_sentences:
        # Return sentence with most keyword matches
        best_sent = max(scored_sentences, key=lambda x: x[1])[0]
        return best_sent.strip()
    
    return "This information is not covered in the lecture. Try asking about: " + ", ".join([s[:30] + "..." for s in sentences[:3]]) + "."

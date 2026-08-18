import re
from functools import lru_cache

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from transformers import pipeline


SUMMARY_MODEL_NAME = "facebook/bart-large-cnn"
MAX_CHARS_PER_CHUNK = 1024
FALLBACK_SUMMARY_PREFIX = "Summary: "


@lru_cache(maxsize=1)
def load_summarizer():
    """Load the pretrained BART summarization model.
    
    Auto-detects GPU (device 0) if available, falls back to CPU (-1).
    """
    device = 0 if (HAS_TORCH and torch.cuda.is_available()) else -1
    
    try:
        return pipeline(
            "summarization",
            model=SUMMARY_MODEL_NAME,
            device=device,
        )
    except Exception:
        return None


def _clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    """Split long transcripts into manageable chunks for CPU-safe summarization."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks = []
    current = ""

    for sentence in sentences:
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

    if not chunks:
        chunks = [cleaned[:max_chars]]

    return chunks


def _fallback_summary(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return "No transcript content was provided."

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]

    selected = sentences[:3]
    summary = " ".join(selected)
    if len(summary) > 500:
        summary = summary[:500].rsplit(" ", 1)[0]

    return f"{FALLBACK_SUMMARY_PREFIX}{summary}"


def _summarize_chunk(summarizer, chunk: str) -> str:
    if not summarizer:
        return chunk[:400].strip()
    
    try:
        # Handle pipeline approach (works for both CPU and GPU)
        if callable(summarizer):
            result = summarizer(
                chunk,
                max_length=130,
                min_length=30,
                do_sample=False,
            )
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].get("summary_text", chunk).strip()
    except Exception:
        pass

    return chunk[:400].strip()


def generate_summary(transcript: str) -> str:
    """Generate a reliable lecture summary using a pretrained Hugging Face model.

    If the model cannot be loaded or inference fails in this environment, we fall back
    to a compressed version of the transcript instead of crashing the app.
    """
    cleaned = _clean_text(transcript)
    if not cleaned:
        return "No transcript content was provided."

    try:
        summarizer = load_summarizer()
        chunks = _split_into_chunks(cleaned)

        chunk_summaries = [
            _summarize_chunk(summarizer, chunk)
            for chunk in chunks[:8]
        ]

        combined = " ".join(chunk_summaries).strip()
        if len(combined) <= 600:
            return combined or cleaned[:400]

        try:
            final_result = summarizer(
                combined,
                max_length=180,
                min_length=50,
                do_sample=False,
            )
            if final_result and isinstance(final_result, list):
                return final_result[0].get("summary_text", combined).strip()
        except Exception:
            pass

        return combined[:600].rstrip()

    except Exception:
        return _fallback_summary(cleaned)


def extract_key_concepts(transcript: str, summary: str = "", max_concepts: int = 5):
    """Derive important concepts in a deterministic, lightweight way from the transcript summary."""
    source = summary or transcript
    cleaned = _clean_text(source)
    if not cleaned:
        return [{"concept": "No content", "explanation": "No transcript content was available."}]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if len(sentence.split()) >= 7
    ]

    if not sentences:
        return [{"concept": "Main Topic", "explanation": "Lecture content is available but not long enough to extract detailed concepts."}]

    keywords = [
        word.lower()
        for word in re.findall(r"[A-Za-z]{4,}", cleaned.lower())
        if word.lower() not in {"there", "about", "which", "these", "those", "using", "lecture", "model", "student", "learn", "from", "with", "into", "that", "have", "this", "will", "their"}
    ]

    from collections import Counter
    word_counts = Counter(keywords)
    top_words = [word for word, _ in word_counts.most_common(12)]

    selected = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in top_words[:6]):
            selected.append(sentence)

    if len(selected) < max_concepts:
        selected.extend(sentences)

    unique = []
    seen = set()
    for sentence in selected:
        key = sentence.lower()
        if key not in seen:
            unique.append(sentence)
            seen.add(key)
        if len(unique) >= max_concepts:
            break

    concepts = []
    for sentence in unique:
        short_title = sentence[:90].rstrip(". ")
        concepts.append({
            "concept": short_title if short_title else "Key Idea",
            "explanation": sentence,
        })

    return concepts[:max_concepts]

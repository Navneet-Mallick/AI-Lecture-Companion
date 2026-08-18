# 🎓 AI Lecture Companion

An AI-powered lecture study assistant that converts recorded lectures into
structured learning material using pretrained AI models.

## Overview

AI Lecture Companion integrates multiple pretrained AI capabilities in a single end-to-end pipeline:

1. Whisper for speech recognition
2. Hugging Face Transformer models for summary and key concepts
3. Hugging Face generation / QA models for quiz and lecture Q&A

The system accepts a lecture recording and converts it into:

- lecture transcript
- AI-generated summary
- key concepts
- multiple-choice quiz
- interactive quiz evaluation
- lecture-grounded question answering

## Architecture

Lecture Audio
    |
    v
Whisper (pretrained speech model)
    |
    v
Transcript
    |
    v
Hugging Face Transformers
    |
    +----> Summary
    |
    +----> Key Concepts
    |
    +----> Quiz Generation
    |
    +----> Lecture Q&A
    |
    v
Streamlit App

## Models Used

- Whisper: speech-to-text
- Facebook BART CNN: summarization
- FLAN-T5: lightweight text generation for quiz generation
- DistilBERT SQuAD: extractive Q&A

## Why these models

These models are pretrained, CPU-friendly, and practical for a Windows laptop academic project. The project demonstrates the pipeline of selecting and combining existing pretrained models rather than training a model from scratch.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## First run note

The first time the app launches, the Hugging Face models may download automatically. This can take a few minutes depending on internet speed and hardware.

## Example workflow

1. Upload a lecture in MP3, MP4, WAV, or M4A format.
2. Whisper transcribes the audio.
3. The Hugging Face summarization model creates a lecture summary.
4. Key concepts are extracted from the transcript and summary.
5. A Hugging Face generation model creates quiz questions.
6. Users answer the quiz and see a score.
7. Users can ask lecture-related questions and get grounded answers from the transcript.

## Notes

- This project uses pretrained inference models only.
- No model training or fine-tuning is performed.
- The app is designed to be practical for CPU-only execution on a laptop.
- Large transcripts are chunked to avoid model input limits.

## Limitations

- Summary quality depends on transcript quality and model size.
- Quiz generation may sometimes produce imperfect or generic questions.
- Very long recordings may take longer to process on CPU.
- Question answering is extractive and grounded in the transcript, so unsupported questions may return a fallback message.

## GPU Acceleration (Optional)

The app runs on **CPU by default** and is optimized for college project submission.

For faster inference on systems with NVIDIA GPUs (optional, post-submission):

```powershell
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision torchaudio -y

# Install GPU-enabled PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**No code changes needed** — the app automatically detects and uses GPU when available.

Expected speedup: ~2-3x faster inference on RTX 3050 and similar GPUs.

## Future improvements

- add a smaller lightweight summarization model option
- improve chunking and concept extraction
- add more robust quiz parsing
- add user-configurable model selection

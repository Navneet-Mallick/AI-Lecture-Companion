# 🎓 AI Lecture Companion

An AI-powered study assistant that converts recorded lectures into structured learning material.

## Features

- 🎤 **Transcription** — Whisper converts lecture audio to text
- 📖 **Summary** — Gemini generates a concise lecture summary
- 🔑 **Key Concepts** — Extracts important topics with explanations
- ❓ **Quiz** — Auto-generated multiple-choice questions with scoring
- 💬 **Q&A** — Ask questions about the lecture, answered from the transcript

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | OpenAI Whisper |
| NLP / Generation | Google Gemini API |
| Frontend | Streamlit |
| Language | Python |

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

## Architecture

```
Lecture Audio
    → Whisper (speech-to-text)
    → Transcript
    → Google Gemini (single API call)
        → Summary
        → Key Concepts
        → Quiz (5 MCQs)
    → Streamlit UI
```

## Project Structure

```
├── app.py              # Main Streamlit application
├── utils/
│   ├── speech.py       # Whisper transcription
│   ├── llm.py          # Study material orchestration
│   └── gemini.py       # Gemini API integration
├── audio/              # Sample audio files
├── requirements.txt    # Python dependencies
└── .env                # API key (not committed)
```

## Notes

- First run downloads the Whisper model (~140 MB) which may take a minute
- Supports MP4, MP3, WAV, and M4A formats
- Max file size: 500 MB

---

**By Navneet Mallick** — AI Semester Project

# 🎓 AI Lecture Companion

An AI-powered study assistant that converts recorded lectures into
structured learning material.

## Overview

AI Lecture Companion uses two major AI capabilities:

1. Speech Recognition using OpenAI Whisper
2. Generative AI using Google Gemini

The system accepts a recorded lecture and converts it into:

- Lecture transcript
- AI-generated summary
- Key concepts
- Multiple-choice quiz
- Interactive quiz evaluation
- Lecture-grounded question answering

## System Architecture

Lecture Video
     |
     v
  Whisper
     |
     v
Transcript
     |
     v
  Gemini
     |
     +----> Summary
     |
     +----> Key Concepts
     |
     +----> Quiz
     |
     +----> Lecture Q&A

## Technologies Used

- Python
- Streamlit
- OpenAI Whisper
- Google Gemini
- PyTorch
- python-dotenv

## AI Capabilities

### 1. Speech Recognition

Whisper converts the audio contained in the lecture recording
into text.

### 2. Generative AI

Gemini analyzes the transcript and generates structured study
material including summaries, concepts and quiz questions.

### 3. Lecture Q&A

Students can ask questions about the lecture. Gemini uses the
lecture transcript as the primary context for generating the answer.

## Features

### Lecture Transcription
Converts recorded lectures into text.

### Summary
Generates a concise summary of the lecture.

### Key Concepts
Extracts important concepts and provides explanations.

### Quiz
Generates multiple-choice questions from the lecture.

### Interactive Quiz
Students can answer questions and receive a score.

### Ask the Lecture
Students can ask questions about the uploaded lecture.

### Download
Users can download the generated transcript and study material.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
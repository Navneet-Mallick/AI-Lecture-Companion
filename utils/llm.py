import json
import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

MODEL_NAME = "gemini-3.5-flash"


def get_client():
    """Create and return the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Please check your .env file."
        )

    return genai.Client(api_key=api_key)


def generate_study_material(transcript: str) -> dict:
    """
    Generate summary, key concepts, and quiz questions
    from a lecture transcript.
    """

    client = get_client()

    prompt = f"""
You are an academic study assistant.

Analyze the following lecture transcript and create
useful study material for a college student.

IMPORTANT RULES:
- Base your response primarily on the transcript.
- Do not invent facts that are not supported by the transcript.
- The transcript may contain speech-recognition errors.
- If a sentence appears obviously corrupted, infer its meaning
  only when the intended meaning is reasonably clear.
- Keep the content educational and concise.

Generate:

1. A concise summary.
2. 5 to 8 important key concepts, each with a short explanation.
3. 5 multiple-choice quiz questions.
4. Each quiz question must have exactly 4 options.
5. Give the correct answer and a short explanation.

Return ONLY valid JSON in this exact structure:

{{
    "summary": "summary text",
    "key_concepts": [
        {{
            "concept": "concept name",
            "explanation": "short explanation"
        }}
    ],
    "quiz": [
        {{
            "question": "question text",
            "options": [
                "option A",
                "option B",
                "option C",
                "option D"
            ],
            "correct_answer": "option text",
            "explanation": "short explanation"
        }}
    ]
}}

LECTURE TRANSCRIPT:
{transcript}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    response_text = response.text.strip()

    # Remove markdown code fences if Gemini adds them.
    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Gemini returned an invalid JSON response."
        ) from error


def ask_lecture(transcript: str, question: str) -> str:
    """
    Answer a student's question using the lecture transcript
    as the primary source of information.
    """

    client = get_client()

    prompt = f"""
You are an AI study assistant.

Answer the student's question using the lecture transcript
provided below.

IMPORTANT RULES:
- Base your answer primarily on the lecture.
- Do not invent information.
- If the answer is not present or cannot reasonably be
  inferred from the lecture, clearly say:
  "This information is not covered in the lecture."
- Give a concise explanation suitable for a college student.
- You may clarify obvious speech-to-text errors when the
  intended meaning is clear.

LECTURE TRANSCRIPT:
{transcript}

STUDENT QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()
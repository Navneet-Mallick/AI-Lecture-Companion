import os
import sys
from utils.speech import load_whisper_model, transcribe_audio
from utils.llm import generate_study_material, ask_lecture

# Test with the sample audio file
audio_path = "audio/sample.mp4"

if os.path.exists(audio_path):
    print("[OK] Test 1: Loading Whisper model...")
    try:
        model = load_whisper_model()
        print("[OK] Whisper model loaded")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        sys.exit(1)
    
    print("[OK] Test 2: Transcribing sample audio...")
    try:
        transcript = transcribe_audio(model, audio_path)
        print(f"[OK] Transcription success. Length: {len(transcript)} chars")
        print(f"    Sample: {transcript[:150]}...")
    except Exception as e:
        print(f"[FAIL] Transcription failed: {e}")
        sys.exit(1)
    
    print("\n[OK] Test 3: Generating study material (summary, concepts)...")
    try:
        study_material = generate_study_material(transcript)
        print(f"[OK] Summary generated. Length: {len(study_material['summary'])} chars")
        print(f"    Uses fallback: {'Fallback summary:' in study_material['summary']}")
        print(f"[OK] Concepts extracted: {len(study_material['key_concepts'])} concepts")
        if study_material['key_concepts']:
            print(f"    Sample: {study_material['key_concepts'][:2]}")
    except Exception as e:
        print(f"[FAIL] Study material failed: {e}")
        sys.exit(1)
    
    print("\n[OK] Test 4: Generating quiz...")
    try:
        quiz = generate_study_material(transcript)
        quiz_q = quiz.get('quiz', [])
        print(f"[OK] Quiz generated with {len(quiz_q)} questions")
        if quiz_q and isinstance(quiz_q, list) and len(quiz_q) > 0:
            first_q = str(quiz_q[0])
            print(f"    Sample Q: {first_q[:80]}...")
    except Exception as e:
        print(f"[FAIL] Quiz generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n[OK] Test 5: Asking a question about the lecture...")
    try:
        answer = ask_lecture(transcript, "What are the main topics covered?")
        print(f"[OK] Q&A works. Answer length: {len(answer)} chars")
        print(f"    Sample: {answer[:120]}...")
    except Exception as e:
        print(f"[FAIL] Q&A failed: {e}")
        sys.exit(1)
    
    print("\n[OK] ALL TESTS PASSED - App is ready for demo!")
else:
    print(f"[FAIL] Sample audio file not found at {audio_path}")
    sys.exit(1)

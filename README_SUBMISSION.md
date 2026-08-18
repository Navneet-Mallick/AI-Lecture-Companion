# AI Lecture Companion - Academic Project

## Overview

**AI Lecture Companion** is an intelligent study material generation system that transforms lecture recordings into comprehensive study resources. The application uses **local Hugging Face Transformer models** for all NLP processing, ensuring reliability, privacy, and offline capability.

### Key Features

✓ **Lecture Transcription** - Automatic speech-to-text using OpenAI Whisper  
✓ **Smart Summarization** - Extract key points using BART  
✓ **Concept Extraction** - Identify and explain main topics  
✓ **Quiz Generation** - Auto-generate multiple-choice questions for self-assessment  
✓ **Interactive Q&A** - Ask questions about lecture content and get contextual answers  
✓ **GPU-Ready** - Automatic GPU acceleration (RTX 3050+) with CPU fallback  

---

## Architecture

### Technology Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| **Transcription** | OpenAI Whisper (base) | Audio→Text conversion |
| **Summarization** | facebook/bart-large-cnn | Extract key lecture points (400MB) |
| **Quiz Generation** | google/flan-t5-base | Generate Q&A pairs (250MB) |
| **Question Answering** | distilbert-base-uncased-distilled-squad | Answer student questions (260MB) |
| **Framework** | Streamlit | Web UI & user interaction |

**Total Model Size**: ~910MB downloaded once, ~2-3GB GPU memory when loaded.

### Workflow

```
Lecture Audio/Video (MP3, MP4, WAV)
         ↓
    [Whisper Transcription]
         ↓
    Full Text Transcript (424+ chars sample)
         ↓
    ┌────────────────────────────────────────┐
    ├→ [BART Summarization]  → Study Summary
    ├→ [Concept Extraction]  → Key Concepts
    ├→ [FLAN-T5 Quiz Gen]    → MCQ Questions
    └→ [DistilBERT Q&A]      → Answer Engine
         ↓
    [Streamlit UI]
         ↓
    Student Dashboard
```

### Model Design Philosophy

✓ **Pretrained Transformers** - All models are pretrained (no training required)  
✓ **Local Inference** - Models run locally for privacy and reliability  
✓ **CPU/GPU Flexible** - Auto-detects GPU; runs on CPU if needed  
✓ **Graceful Fallbacks** - Keyword extraction replaces model output if unavailable  
✓ **Production-Ready** - Error handling, validation, and logging throughout  

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- 16GB RAM recommended (works on 8GB with CPU-only)
- RTX 3050 or similar GPU (optional but recommended)

### Step 1: Clone/Setup Project

```bash
cd c:\Users\YourName\Desktop\AI-Lecture-Companion
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: GPU Support (Optional, Install Later)

For CUDA 12.1 GPU acceleration on RTX 3050:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8 (alternative):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

Open browser to: **http://localhost:8501**

---

## Usage

### Demo Video (30-60 seconds)

1. **Upload** a lecture recording (MP3, MP4, WAV) or use sample
2. **Wait** for transcription (20-40 seconds depending on length)
3. **View** summary, key concepts, and quiz
4. **Ask** questions in the Q&A section
5. **Download** study materials as needed

### Sample Inputs

Test with the included sample lecture:

```
audio/sample.mp4  →  "Artificial Intelligence & Machine Learning"
```

### Expected Outputs

**Summary**: 300-500 word condensed version of lecture  
**Concepts**: 3-5 key topics with explanations  
**Quiz**: 5 multiple-choice questions  
**Q&A**: Contextual answers from lecture content  

---

## File Structure

```
AI-Lecture-Companion/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
├── utils/
│   ├── __init__.py
│   ├── speech.py            # Whisper transcription
│   ├── llm.py               # Main orchestrator
│   ├── summarizer.py        # BART summarization + fallback
│   ├── quiz.py              # FLAN-T5 quiz generation
│   └── qa.py                # DistilBERT question answering
│
├── audio/
│   └── sample.mp4           # Example lecture
│
└── test_*.py                # Testing scripts
```

---

## Model Compatibility & Performance

### RTX 3050 (Recommended)

| Model | Load Time | Inference | Total |
|-------|-----------|-----------|-------|
| Whisper | 15s | 2-5s/min audio | ~20s/10min lecture |
| BART Summary | 8s | 2-3s | Fast ✓ |
| FLAN-T5 Quiz | 6s | 2-3s | Fast ✓ |
| DistilBERT QA | 4s | 1-2s | Very Fast ✓ |

### CPU-Only (Works, Slower)

- Models load on CPU (Intel i7-HS can handle 1 model concurrently)
- Each operation takes 2-5x longer
- Still viable for college project demonstration
- **Upgrade to GPU after submission for better performance**

### Memory Usage

- **Idle**: ~500MB
- **One Model Loaded**: ~1.5-2GB
- **All Models Loaded**: ~4-5GB (RTX 3050: 6GB VRAM available)

---

## Limitations & Fallbacks

| Issue | Fallback Strategy | User Impact |
|-------|-------------------|-------------|
| Summarizer model unavailable | Extract first 3 sentences | Still get study summary |
| Quiz generation fails | Templated questions from transcript | Still get 5 questions |
| QA model unavailable | Keyword extraction from transcript | Still get relevant answers |
| No GPU/CUDA | Auto-switch to CPU inference | Slower but functional |

---

## Testing & Validation

### Run All Tests

```bash
.\.venv\Scripts\python test_full_pipeline.py
```

### Expected Output

```
[OK] Test 1: Loading Whisper model...
[OK] Test 2: Transcribing sample audio...
[OK] Test 3: Generating study material...
[OK] Test 4: Generating quiz...
[OK] Test 5: Asking questions...
[OK] ALL TESTS PASSED
```

### Validate Code Quality

```bash
.\.venv\Scripts\python -m py_compile app.py utils/*.py
```

---

## Academic Project Features for Submission

### ✓ Core Competencies Demonstrated

1. **Transfer Learning**
   - Using 4 pretrained transformer models without training
   - Justification: Efficient, industry-standard approach

2. **NLP Pipeline Integration**
   - Connects multiple AI models in production sequence
   - Demonstrates understanding of model chaining

3. **Error Handling & Robustness**
   - Fallback summaries prevent crashes
   - Graceful degradation if models unavailable

4. **Performance Optimization**
   - GPU/CPU auto-detection
   - Model caching with `@lru_cache`
   - Chunking for long texts

5. **User-Centric Design**
   - Streamlit UI for non-technical users
   - Clear error messages and status updates
   - Demo-friendly

### ✓ Why No API Keys Required

- **Privacy**: All processing local (no cloud dependency)
- **Reliability**: Works offline, no API quota issues
- **Academic**: Demonstrates ML infrastructure, not cloud APIs
- **Submission-Safe**: No authentication failures during demo

---

## Troubleshooting

### "CUDA not available" on Windows

**Solution**: This is normal for CPU-only PyTorch. Models still work on CPU.  
**To Fix**: See GPU Installation section above.

### Port 8501 Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Model Download Timeout

Models cache to: `C:\Users\<YourName>\.cache\huggingface\hub\`

If slow:
1. Check internet connection
2. Restart after 5 minutes
3. Model will cache permanently after first download

### Audio File Not Recognized

Supported formats: **MP3, MP4, WAV, M4A**

---

## Performance Tips

### First Run (Slower)

- Models download from Hugging Face (5-10 min total)
- After first use, models cache locally
- Subsequent runs much faster

### Optimize for Demo

1. Pre-load models before recording demo video
2. Use short test audio (1-2 minutes)
3. Ensure GPU CUDA is installed for RTX 3050
4. Close other applications to free RAM

### Production Deployment

For serving many students:
- Consider model quantization
- Use model serving frameworks (Hugging Face Transformers + FastAPI)
- Implement batching for multiple uploads

---

## Future Enhancements (Beyond Scope)

- [ ] Fine-tune models for specific domains (STEM, Humanities)
- [ ] Real-time streaming transcription
- [ ] Export to PDF/DOCX
- [ ] Multi-language support
- [ ] Model quantization for faster inference
- [ ] Spoken Q&A (audio input)

---

## References & Citations

**Models Used**:
- Whisper: https://github.com/openai/whisper
- BART: https://huggingface.co/facebook/bart-large-cnn
- FLAN-T5: https://huggingface.co/google/flan-t5-base
- DistilBERT: https://huggingface.co/distilbert-base-uncased-distilled-squad

**Frameworks**:
- Streamlit: https://streamlit.io
- Transformers: https://huggingface.co/transformers/
- PyTorch: https://pytorch.org

---

## License & Academic Use

This project is provided for educational purposes. All models are open-source and freely available for academic and commercial use under their respective licenses.

**For Submission**: This project demonstrates practical AI engineering using industry-standard tools and techniques.

---

**Last Updated**: 2026-08-18  
**Status**: Demo-Ready for College Submission

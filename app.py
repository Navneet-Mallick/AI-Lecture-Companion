import os
import tempfile

import streamlit as st

from utils.speech import load_whisper_model, transcribe_audio
from utils.llm import (
    generate_summary_only,
    generate_concepts_only,
    generate_quiz_only,
    ask_lecture,
)


# Page Configuration
st.set_page_config(
    page_title="AI Lecture Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-title { font-size: 42px; font-weight: 700; margin-bottom: 5px; }
.subtitle { font-size: 18px; opacity: 0.75; margin-bottom: 25px; }
.feature-card {
    padding: 20px; border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25); min-height: 140px;
}
.feature-title { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
.section-title { font-size: 26px; font-weight: 600; margin-top: 10px; margin-bottom: 15px; }
.footer { text-align: center; opacity: 0.6; padding: 30px 0 10px 0; font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("🎓 AI Lecture Companion")
    st.divider()

    st.markdown("### 🤖 Technologies Used")
    st.markdown("""
**Whisper** — Speech recognition model for
converting lecture audio into text.

**Google Gemini** — Generative AI for summaries,
key concepts, quiz questions, and Q&A.
""")

    st.divider()
    st.markdown("### ✨ Features")
    st.markdown("""
📝 Lecture Transcription  
📖 Summary  
🔑 Key Concepts  
❓ Quiz  
💬 Lecture Q&A  
📥 Download Study Material
""")

    st.divider()
    st.markdown("""
**Project:** AI / Generative AI  
**Domain:** Education Technology  
**By:** Navneet Mallick
""")


# Header
st.markdown('<div class="main-title">🎓 AI Lecture Companion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Transform recorded lectures into intelligent study material — By Navneet Mallick'
    '</div>',
    unsafe_allow_html=True
)

# Feature Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="feature-card">
        <div class="feature-title">🎤 Transcribe</div>
        Convert lecture audio into text using Whisper.
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="feature-card">
        <div class="feature-title">📖 Summarize</div>
        Get a concise lecture summary.
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="feature-card">
        <div class="feature-title">❓ Practice</div>
        Auto-generated quiz from the lecture.
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="feature-card">
        <div class="feature-title">💬 Ask</div>
        Ask questions about the lecture.
    </div>""", unsafe_allow_html=True)

st.divider()


# Whisper Model (cached)
@st.cache_resource
def get_whisper_model():
    return load_whisper_model()


# File Upload
st.markdown('<div class="section-title">📁 Upload Lecture</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a lecture recording",
    type=["mp4", "mp3", "wav", "m4a"],
    help="Supported formats: MP4, MP3, WAV, M4A"
)

MAX_FILE_SIZE_MB = 500

if uploaded_file is None:
    st.info("📤 Upload a lecture recording to get started.")
    st.stop()

if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
    st.error(f"❌ File too large. Max size: {MAX_FILE_SIZE_MB} MB.")
    st.stop()


# Process Lecture
st.success(f"📄 Selected: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")

if st.button("🚀 Process Lecture", type="primary", use_container_width=True):

    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        # Stage 1: Transcription
        progress = st.progress(0, text="🎤 Transcribing lecture with Whisper...")
        with st.spinner("🎤 Transcribing..."):
            model = get_whisper_model()
            transcript = transcribe_audio(model, temp_path)

        if not transcript:
            st.error("❌ No speech detected in this file.")
            st.stop()

        st.session_state["transcript"] = transcript
        progress.progress(25, text="✅ Transcription complete!")
        st.success("📝 Transcript ready!")

        # Stage 2: Summary
        progress.progress(30, text="📖 Generating summary...")
        with st.spinner("📖 Generating summary..."):
            summary = generate_summary_only(transcript)

        st.session_state["summary"] = summary
        progress.progress(50, text="✅ Summary generated!")

        # Stage 3: Key Concepts
        progress.progress(55, text="🔑 Extracting key concepts...")
        with st.spinner("🔑 Extracting key concepts..."):
            key_concepts = generate_concepts_only(transcript, summary)

        st.session_state["key_concepts"] = key_concepts
        progress.progress(75, text="✅ Key concepts extracted!")

        # Stage 4: Quiz
        progress.progress(80, text="❓ Generating quiz...")
        with st.spinner("❓ Generating quiz questions..."):
            quiz = generate_quiz_only(summary, transcript, key_concepts)

        st.session_state["quiz"] = quiz
        progress.progress(100, text="✅ All done!")

        st.session_state["answer"] = ""
        st.success("🎉 Lecture processed successfully!")

    except Exception as e:
        st.error("❌ Something went wrong while processing the lecture.")
        st.exception(e)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Display Results
if "transcript" in st.session_state:
    transcript = st.session_state["transcript"]

    st.divider()
    st.markdown('<div class="section-title">📚 Study Material</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Transcript", "📚 Summary & Concepts", "❓ Quiz", "💬 Ask Lecture"
    ])

    # Tab 1: Transcript
    with tab1:
        st.subheader("📝 Lecture Transcript")
        st.text_area("Transcript", transcript, height=450, label_visibility="collapsed")
        st.download_button(
            "⬇️ Download Transcript",
            data=transcript,
            file_name="lecture_transcript.txt",
            mime="text/plain"
        )

    # Tab 2: Summary & Key Concepts
    with tab2:
        summary = st.session_state.get("summary", "")
        key_concepts = st.session_state.get("key_concepts", [])

        if summary:
            st.subheader("📖 Summary")
            st.write(summary)
        else:
            st.info("Summary is being generated...")

        if key_concepts:
            st.divider()
            st.subheader("🔑 Key Concepts")
            for i, concept in enumerate(key_concepts, 1):
                with st.expander(f"{i}. {concept['concept']}", expanded=True):
                    st.write(concept["explanation"])

            # Download
            study_text = (
                "AI LECTURE COMPANION\n"
                "====================\n\n"
                "SUMMARY\n-------\n\n"
                f"{summary}\n\n"
                "KEY CONCEPTS\n------------\n\n"
            )
            for concept in key_concepts:
                study_text += f"• {concept['concept']}\n  {concept['explanation']}\n\n"

            st.download_button(
                "⬇️ Download Study Material",
                data=study_text,
                file_name="lecture_study_material.txt",
                mime="text/plain"
            )

    # Tab 3: Quiz
    with tab3:
        quiz = st.session_state.get("quiz", [])

        if not quiz:
            st.info("Quiz is being generated...")
        else:
            st.subheader("🧠 Test Your Knowledge")

            if "quiz_submitted" not in st.session_state:
                st.session_state["quiz_submitted"] = False
            if "quiz_score" not in st.session_state:
                st.session_state["quiz_score"] = 0

            for i, q in enumerate(quiz, 1):
                st.markdown(f"### {i}. {q['question']}")
                selected = st.radio(
                    "Choose an answer:",
                    q["options"],
                    key=f"quiz_question_{i}",
                    index=None
                )
                st.session_state[f"selected_answer_{i}"] = selected
                st.divider()

            if st.button("📊 Submit Quiz", type="primary", use_container_width=True, key="submit_quiz"):
                score = 0
                unanswered = 0
                for i, q in enumerate(quiz, 1):
                    ans = st.session_state.get(f"selected_answer_{i}")
                    if ans is None:
                        unanswered += 1
                    elif ans == q["correct_answer"]:
                        score += 1

                st.session_state["quiz_score"] = score
                st.session_state["quiz_unanswered"] = unanswered
                st.session_state["quiz_submitted"] = True

            if st.session_state.get("quiz_submitted"):
                score = st.session_state["quiz_score"]
                unanswered = st.session_state.get("quiz_unanswered", 0)
                total = len(quiz)
                percentage = (score / total) * 100

                st.divider()
                st.subheader("📊 Result")

                c1, c2, c3 = st.columns(3)
                c1.metric("Score", f"{score}/{total}")
                c2.metric("Percentage", f"{percentage:.0f}%")
                c3.metric("Unanswered", unanswered)

                if percentage >= 80:
                    st.success("🎉 Excellent! Strong understanding of the lecture.")
                elif percentage >= 60:
                    st.info("👍 Good job! Review concepts to strengthen understanding.")
                else:
                    st.warning("📚 Consider reviewing the summary and concepts.")

                st.divider()
                st.subheader("📋 Answer Review")

                for i, q in enumerate(quiz, 1):
                    ans = st.session_state.get(f"selected_answer_{i}")
                    correct = q["correct_answer"]

                    st.markdown(f"**{i}. {q['question']}**")
                    if ans is None:
                        st.warning("⚠️ Not answered")
                    elif ans == correct:
                        st.success("✅ Correct")
                    else:
                        st.error("❌ Incorrect")
                        st.write(f"Your answer: {ans}")

                    st.write(f"Correct answer: **{correct}**")
                    st.write(f"Explanation: {q['explanation']}")
                    st.divider()

                if st.button("🔄 Try Again", key="retry_quiz"):
                    st.session_state["quiz_submitted"] = False
                    st.session_state["quiz_score"] = 0
                    st.session_state["quiz_unanswered"] = 0
                    for i in range(1, len(quiz) + 1):
                        st.session_state.pop(f"selected_answer_{i}", None)
                    st.rerun()

    # Tab 4: Ask Lecture
    with tab4:
        st.subheader("💬 Ask the Lecture")
        st.write("Ask any question — answered using the lecture transcript as context.")
        st.info("💡 Example: What are the main topics covered?")

        question = st.text_input(
            "Your question",
            placeholder="e.g. What is supervised learning?",
            key="question_input"
        )

        if st.button("🔎 Ask", type="primary", key="ask_button"):
            if not question.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_lecture(transcript, question)
                        st.session_state["answer"] = answer
                    except Exception as e:
                        st.error("❌ Unable to answer the question.")
                        st.exception(e)

        if st.session_state.get("answer"):
            st.divider()
            st.subheader("💡 Answer")
            st.write(st.session_state["answer"])


# How It Works
st.divider()
st.markdown('<div class="section-title">⚙️ How It Works</div>', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("### 1️⃣ Upload\nUpload a recorded lecture in MP4, MP3, WAV or M4A format.")
with s2:
    st.markdown("### 2️⃣ Process\nWhisper transcribes audio, then Gemini generates study material.")
with s3:
    st.markdown("### 3️⃣ Study\nGet summaries, concepts, quizzes and interactive Q&A.")

# Footer
st.markdown("""
<div class="footer">
    🎓 AI Lecture Companion | AI Semester Project — Navneet Mallick
</div>
""", unsafe_allow_html=True)

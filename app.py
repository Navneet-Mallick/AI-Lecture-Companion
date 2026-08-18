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

st.success(f"📄 Selected: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")

if st.button("🚀 Process Lecture", type="primary", use_container_width=True):

    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    try:
        # Stage 1: Transcription
        status = st.status("Processing lecture...", expanded=True)
        status.write("🎤 **Stage 1/4:** Transcribing with Whisper...")

        model = get_whisper_model()
        transcript = transcribe_audio(model, temp_path)

        if not transcript:
            st.error("❌ No speech detected in this file.")
            st.stop()

        status.write("✅ Transcription complete!")
        st.session_state["transcript"] = transcript

        # Show transcript immediately
        with st.expander("📝 Transcript", expanded=True):
            st.text_area("transcript_preview", transcript, height=300, label_visibility="collapsed")

        # Stage 2: Summary
        status.write("📖 **Stage 2/4:** Generating summary with Gemini...")
        summary = generate_summary_only(transcript)
        status.write("✅ Summary generated!")
        st.session_state["summary"] = summary

        # Show summary immediately
        with st.expander("📖 Summary", expanded=True):
            st.write(summary)

        # Stage 3: Key Concepts
        status.write("🔑 **Stage 3/4:** Extracting key concepts...")
        key_concepts = generate_concepts_only(transcript, summary)
        status.write("✅ Key concepts extracted!")
        st.session_state["key_concepts"] = key_concepts

        # Show concepts immediately
        with st.expander("🔑 Key Concepts", expanded=True):
            for i, concept in enumerate(key_concepts, 1):
                st.markdown(f"**{i}. {concept['concept']}**")
                st.write(concept["explanation"])
                st.write("")

        # Stage 4: Quiz
        status.write("❓ **Stage 4/4:** Generating quiz questions...")
        quiz = generate_quiz_only(summary, transcript, key_concepts)
        status.write("✅ Quiz generated!")
        st.session_state["quiz"] = quiz
        st.session_state["answer"] = ""

        status.update(label="🎉 Processing complete!", state="complete", expanded=False)

    except Exception as e:
        st.error("❌ Something went wrong while processing the lecture.")
        st.exception(e)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Display full results (after processing or on page reload)
if "transcript" in st.session_state:
    transcript = st.session_state["transcript"]
    summary = st.session_state.get("summary", "")
    key_concepts = st.session_state.get("key_concepts", [])
    quiz = st.session_state.get("quiz", [])

    st.divider()
    st.markdown('<div class="section-title">📚 Study Material</div>', unsafe_allow_html=True)

    # Transcript
    st.subheader("📝 Transcript")
    st.text_area("Full Transcript", transcript, height=300, label_visibility="collapsed")
    st.download_button(
        "⬇️ Download Transcript",
        data=transcript,
        file_name="lecture_transcript.txt",
        mime="text/plain"
    )

    # Summary
    if summary:
        st.divider()
        st.subheader("📖 Summary")
        st.write(summary)

    # Key Concepts
    if key_concepts:
        st.divider()
        st.subheader("🔑 Key Concepts")
        for i, concept in enumerate(key_concepts, 1):
            with st.expander(f"{i}. {concept['concept']}", expanded=True):
                st.write(concept["explanation"])

        # Download study material
        study_text = (
            "AI LECTURE COMPANION\n====================\n\n"
            f"SUMMARY\n-------\n\n{summary}\n\n"
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

    # Quiz
    if quiz:
        st.divider()
        st.subheader("❓ Quiz")

        if "quiz_submitted" not in st.session_state:
            st.session_state["quiz_submitted"] = False

        for i, q in enumerate(quiz, 1):
            st.markdown(f"**{i}. {q['question']}**")
            selected = st.radio(
                "Choose:",
                q["options"],
                key=f"quiz_q_{i}",
                index=None
            )
            st.session_state[f"sel_{i}"] = selected
            st.write("")

        if st.button("📊 Submit Quiz", type="primary", use_container_width=True, key="submit_quiz"):
            score = 0
            unanswered = 0
            for i, q in enumerate(quiz, 1):
                ans = st.session_state.get(f"sel_{i}")
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
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{score}/{total}")
            c2.metric("Percentage", f"{percentage:.0f}%")
            c3.metric("Unanswered", unanswered)

            if percentage >= 80:
                st.success("🎉 Excellent!")
            elif percentage >= 60:
                st.info("👍 Good job!")
            else:
                st.warning("📚 Review the material and try again.")

            st.divider()
            for i, q in enumerate(quiz, 1):
                ans = st.session_state.get(f"sel_{i}")
                correct = q["correct_answer"]
                st.markdown(f"**{i}. {q['question']}**")
                if ans is None:
                    st.warning("Not answered")
                elif ans == correct:
                    st.success(f"✅ {ans}")
                else:
                    st.error(f"❌ {ans}")
                st.write(f"Correct: **{correct}** — {q['explanation']}")
                st.write("")

            if st.button("🔄 Try Again", key="retry"):
                st.session_state["quiz_submitted"] = False
                for i in range(1, len(quiz) + 1):
                    st.session_state.pop(f"sel_{i}", None)
                st.rerun()

    # Ask Lecture
    st.divider()
    st.subheader("💬 Ask the Lecture")
    question = st.text_input("Your question", placeholder="e.g. What is supervised learning?", key="q_input")

    if st.button("🔎 Ask", type="primary", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = ask_lecture(transcript, question)
                    st.session_state["answer"] = answer
                except Exception as e:
                    st.error("❌ Unable to answer.")
                    st.exception(e)

    if st.session_state.get("answer"):
        st.write("")
        st.markdown(f"**💡 Answer:** {st.session_state['answer']}")


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

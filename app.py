import os
import tempfile

import streamlit as st

from utils.speech import load_whisper_model, transcribe_audio
from utils.llm import generate_study_material, ask_lecture


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Lecture Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    /* Feature cards */
    .feature-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        min-height: 140px;
    }

    .feature-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Section headings */
    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.6;
        padding: 30px 0 10px 0;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🎓 AI Lecture")
    st.title("Companion")

    st.divider()

    st.markdown("### 🤖 AI Technologies")

    st.markdown(
        """
        **Whisper**
        
        Speech recognition model used to
        convert lecture audio into text.
        
        **Gemini**
        
        Generative AI model used to create
        study material and answer questions.
        """
    )

    st.divider()

    st.markdown("### ✨ Features")

    st.markdown(
        """
        📝 Lecture Transcription  
        📖 AI Summary  
        🔑 Key Concepts  
        ❓ AI Generated Quiz  
        💬 Lecture Q&A  
        📥 Study Material Download
        """
    )

    st.divider()

    st.markdown(
        """
        **Project Type:**  
        AI / Generative AI

        **Application:**  
        Education Technology
        """
    )


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🎓 AI Lecture Companion</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Transform recorded lectures into intelligent study material '
    'using Artificial Intelligence.'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# FEATURE CARDS
# ==================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">🎤 Transcribe</div>
            Convert lecture audio into text
            using Whisper AI.
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">📖 Summarize</div>
            Get a concise AI-generated
            lecture summary.
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">❓ Practice</div>
            Automatically generate
            questions from the lecture.
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-title">💬 Ask</div>
            Ask questions about the
            lecture using AI.
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ==================================================
# WHISPER MODEL
# ==================================================

@st.cache_resource
def get_whisper_model():

    return load_whisper_model()


# ==================================================
# FILE UPLOAD
# ==================================================

st.markdown(
    '<div class="section-title">📁 Upload Lecture</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a lecture recording",
    type=["mp4", "mp3", "wav", "m4a"],
    help="Supported formats: MP4, MP3, WAV and M4A."
)

MAX_FILE_SIZE_MB = 500

if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:

    st.error(
        f"❌ File is too large. "
        f"Please upload a file smaller than "
        f"{MAX_FILE_SIZE_MB} MB."
    )

    st.stop()
# ==================================================
# PROCESS LECTURE
# ==================================================

if uploaded_file is not None:

    st.success(
        f"📄 Selected file: **{uploaded_file.name}**"
    )

    st.write(
        f"File size: "
        f"{uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    if st.button(
        "🚀 Process Lecture",
        type="primary",
        use_container_width=True
    ):

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name


        try:

            # ======================================
            # WHISPER
            # ======================================

            progress = st.progress(
                0,
                text="Starting lecture processing..."
            )

            with st.spinner(
                "🎤 Whisper is transcribing the lecture..."
            ):

                model = get_whisper_model()

                transcript = transcribe_audio(
                    model,
                    temp_path
                )

            progress.progress(
                50,
                text="✅ Transcription complete. Generating study material..."
            )


            if not transcript:

                st.error(
                    "❌ No speech could be detected in this file."
                )

                st.stop()


            # ======================================
            # GEMINI
            # ======================================

            with st.spinner(
                "🤖 Gemini is creating your study material..."
            ):

                study_material = generate_study_material(
                    transcript
                )


            progress.progress(
                100,
                text="✅ Lecture processing complete!"
            )


            st.success(
                "🎉 Your lecture has been successfully processed!"
            )


            # ======================================
            # SAVE RESULTS
            # ======================================

            st.session_state["transcript"] = transcript

            st.session_state["study_material"] = (
                study_material
            )

            st.session_state["answer"] = ""

            st.session_state["question"] = ""


        except Exception as e:

            st.error(
                "❌ Something went wrong while processing "
                "the lecture."
            )

            st.exception(e)


        finally:

            if os.path.exists(temp_path):

                os.remove(temp_path)


# ==================================================
# DISPLAY RESULTS
# ==================================================

if "transcript" in st.session_state:

    transcript = st.session_state["transcript"]

    study_material = (
        st.session_state["study_material"]
    )

    st.divider()

    st.markdown(
        '<div class="section-title">📚 Your Study Material</div>',
        unsafe_allow_html=True
    )


    # ==================================================
    # TABS
    # ==================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📝 Transcript",
            "📚 Study Material",
            "❓ Quiz",
            "💬 Ask Lecture"
        ]
    )


    # ==================================================
    # TRANSCRIPT
    # ==================================================

    with tab1:

        st.subheader(
            "📝 Lecture Transcript"
        )

        st.caption(
            "Generated using Whisper Speech Recognition AI"
        )

        st.text_area(
            "Transcript",
            transcript,
            height=450,
            label_visibility="collapsed"
        )

        st.download_button(
            "⬇️ Download Transcript",
            data=transcript,
            file_name="lecture_transcript.txt",
            mime="text/plain"
        )


    # ==================================================
    # STUDY MATERIAL
    # ==================================================

    with tab2:

        st.subheader(
            "📖 Summary"
        )

        st.write(
            study_material["summary"]
        )

        st.divider()

        st.subheader(
            "🔑 Key Concepts"
        )

        for index, concept in enumerate(
            study_material["key_concepts"],
            start=1
        ):

            with st.expander(
                f"{index}. {concept['concept']}",
                expanded=True
            ):

                st.write(
                    concept["explanation"]
                )


        # ------------------------------------------
        # Prepare download
        # ------------------------------------------

        study_text = (
            "AI LECTURE COMPANION\n"
            "====================\n\n"
            "SUMMARY\n"
            "-------\n\n"
            f"{study_material['summary']}\n\n"
            "KEY CONCEPTS\n"
            "------------\n\n"
        )

        for concept in study_material["key_concepts"]:

            study_text += (
                f"{concept['concept']}\n"
                f"{concept['explanation']}\n\n"
            )


        st.download_button(
            "⬇️ Download Study Material",
            data=study_text,
            file_name="lecture_study_material.txt",
            mime="text/plain"
        )

    # ==================================================
# QUIZ
# ==================================================

with tab3:

    st.subheader("🧠 Test Your Knowledge")

    st.caption(
        "Answer the questions generated from your lecture."
    )

    quiz = study_material["quiz"]

    # Initialize quiz state
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False

    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0

    # ----------------------------------------------
    # Display questions
    # ----------------------------------------------

    for index, question_data in enumerate(
        quiz,
        start=1
    ):

        st.markdown(
            f"### {index}. {question_data['question']}"
        )

        # Radio button for each question
        selected_answer = st.radio(
            "Choose an answer:",
            question_data["options"],
            key=f"quiz_question_{index}",
            index=None
        )

        # Store selected answer
        st.session_state[
            f"selected_answer_{index}"
        ] = selected_answer

        st.divider()


    # ----------------------------------------------
    # Submit quiz
    # ----------------------------------------------

    if st.button(
        "📊 Submit Quiz",
        type="primary",
        use_container_width=True
    ):

        score = 0
        unanswered = 0

        for index, question_data in enumerate(
            quiz,
            start=1
        ):

            selected_answer = st.session_state.get(
                f"selected_answer_{index}"
            )

            if selected_answer is None:

                unanswered += 1

            elif selected_answer == question_data[
                "correct_answer"
            ]:

                score += 1


        st.session_state["quiz_score"] = score
        st.session_state["quiz_unanswered"] = unanswered
        st.session_state["quiz_submitted"] = True


    # ----------------------------------------------
    # Display score
    # ----------------------------------------------

    if st.session_state.get("quiz_submitted"):

        score = st.session_state["quiz_score"]

        unanswered = st.session_state.get(
            "quiz_unanswered",
            0
        )

        total = len(quiz)

        percentage = (score / total) * 100


        st.divider()

        st.subheader("📊 Your Result")

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                "Score",
                f"{score}/{total}"
            )


        with result_col2:

            st.metric(
                "Percentage",
                f"{percentage:.0f}%"
            )


        with result_col3:

            st.metric(
                "Unanswered",
                unanswered
            )


        # ------------------------------------------
        # Result message
        # ------------------------------------------

        if percentage >= 80:

            st.success(
                "🎉 Excellent! You have a strong "
                "understanding of the lecture."
            )

        elif percentage >= 60:

            st.info(
                "👍 Good job! Review the key concepts "
                "to strengthen your understanding."
            )

        else:

            st.warning(
                "📚 Consider reviewing the summary "
                "and key concepts before trying again."
            )


        # ------------------------------------------
        # Detailed answers
        # ------------------------------------------

        st.divider()

        st.subheader("📋 Answer Review")

        for index, question_data in enumerate(
            quiz,
            start=1
        ):

            selected_answer = st.session_state.get(
                f"selected_answer_{index}"
            )

            correct_answer = question_data[
                "correct_answer"
            ]

            st.markdown(
                f"**{index}. "
                f"{question_data['question']}**"
            )

            if selected_answer is None:

                st.warning(
                    "⚠️ Not answered"
                )

            elif selected_answer == correct_answer:

                st.success(
                    "✅ Correct"
                )

            else:

                st.error(
                    "❌ Incorrect"
                )

                st.write(
                    f"Your answer: {selected_answer}"
                )

            st.write(
                f"Correct answer: **{correct_answer}**"
            )

            st.write(
                f"Explanation: "
                f"{question_data['explanation']}"
            )

            st.divider()


        # ------------------------------------------
        # Try again
        # ------------------------------------------

        if st.button(
            "🔄 Try Quiz Again"
        ):

            st.session_state["quiz_submitted"] = False
            st.session_state["quiz_score"] = 0
            st.session_state["quiz_unanswered"] = 0

            for index in range(1, len(quiz) + 1):

                st.session_state.pop(
                    f"selected_answer_{index}",
                    None
                )

            st.rerun()
   
    # ==================================================
    # ASK LECTURE
    # ==================================================

    with tab4:

        st.subheader(
            "💬 Ask the Lecture"
        )

        st.write(
            "Ask questions about the uploaded lecture. "
            "The AI uses the lecture transcript as its "
            "primary source."
        )

        st.info(
            "💡 Example: What is supervised learning?"
        )

        question = st.text_input(
            "Your question",
            placeholder="e.g. What is supervised learning?",
            key="question_input"
        )

        if st.button(
            "🔎 Ask",
            type="primary",
            key="ask_button"
        ):

            if not question.strip():

                st.warning(
                    "⚠️ Please enter a question."
                )

            else:

                with st.spinner(
                    "🤖 Analyzing the lecture..."
                ):

                    try:

                        answer = ask_lecture(
                            transcript,
                            question
                        )

                        st.session_state["answer"] = answer

                    except Exception as e:

                        st.error(
                            "❌ Unable to answer the question."
                        )

                        st.exception(e)


        if st.session_state.get("answer"):

            st.divider()

            st.subheader(
                "💡 AI Answer"
            )

            st.write(
                st.session_state["answer"]
            )


# ==================================================
# HOW IT WORKS
# ==================================================

st.divider()

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3)


with step1:

    st.markdown(
        """
        ### 1️⃣ Upload

        Upload a recorded lecture in
        MP4, MP3, WAV or M4A format.
        """
    )


with step2:

    st.markdown(
        """
        ### 2️⃣ AI Processing

        Whisper converts speech to text,
        then Gemini analyzes the transcript.
        """
    )


with step3:

    st.markdown(
        """
        ### 3️⃣ Study

        Get summaries, concepts, quizzes
        and interactive lecture Q&A.
        """
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        🎓 AI Lecture Companion |
        Built as an AI Semester Project
    </div>
    """,
    unsafe_allow_html=True
)
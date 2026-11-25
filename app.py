
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from datetime import datetime
from detector import analyze_audio


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_filename" not in st.session_state:
    st.session_state.analysis_filename = None


# ============================================================
# DARK THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top right, #142d50, transparent 35%),
        linear-gradient(135deg, #07111f, #0b1830 55%, #101e38);
    color: #e5edf8;
}

[data-testid="stSidebar"] {
    background: #091426;
    border-right: 1px solid #203653;
}

h1, h2, h3, h4 {
    color: #e5f3ff !important;
}

p, label, span {
    color: #cbd5e1;
}

.stButton button {
    background: linear-gradient(90deg, #0891b2, #0e7490);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px;
    font-weight: 700;
}

.stButton button:hover {
    background: linear-gradient(90deg, #06b6d4, #0891b2);
    color: white;
}

.stDownloadButton button {
    background: #164e63;
    color: white;
    border: 1px solid #0e7490;
    border-radius: 9px;
}

div[data-testid="stMetric"] {
    background: #10243e;
    border: 1px solid #263f5f;
    padding: 15px;
    border-radius: 12px;
}

hr {
    border-color: #29415e;
}

[data-testid="stFileUploader"] {
    background: rgba(16, 36, 62, 0.7);
    border: 1px solid #29415e;
    border-radius: 12px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SCORE ANALYSIS
# ============================================================

def get_score_strength(result):

    ai_score = float(result["ai_score"])
    human_score = float(result["human_score"])

    margin = abs(ai_score - human_score) * 100

    if margin >= 30:
        strength = "Strong model preference"
        explanation = (
            "The model gives a clear preference to one class."
        )

    elif margin >= 15:
        strength = "Moderate model preference"
        explanation = (
            "The model favours one class, but the scores are "
            "not widely separated."
        )

    else:
        strength = "Close scores"
        explanation = (
            "The model scores are relatively close. "
            "Its prediction should be treated cautiously."
        )

    return {
        "margin": margin,
        "strength": strength,
        "explanation": explanation
    }


# ============================================================
# MODEL RESULT CARD
# ============================================================

def display_model_result(model_name, description, result):

    ai_score = float(result["ai_score"]) * 100
    human_score = float(result["human_score"]) * 100

    prediction = result["prediction"]

    score_info = get_score_strength(result)

    with st.container(border=True):

        st.subheader(model_name)
        st.caption(description)

        if prediction == "AI Generated":
            st.error("Prediction: AI Generated")
        else:
            st.success("Prediction: Human")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "AI Voice Score",
                f"{ai_score:.2f}%"
            )

        with col2:
            st.metric(
                "Human Voice Score",
                f"{human_score:.2f}%"
            )

        st.write("AI Voice Score")
        st.progress(max(0.0, min(1.0, ai_score / 100)))

        st.write("Human Voice Score")
        st.progress(max(0.0, min(1.0, human_score / 100)))

        st.divider()

        st.markdown("**Model Score Strength**")

        st.info(
            f"{score_info['strength']}\n\n"
            f"Score difference: {score_info['margin']:.2f} percentage points"
        )

        st.caption(score_info["explanation"])


# ============================================================
# ADVANCED OVERALL ASSESSMENT
# ============================================================

def display_assessment(results):

    cnn = results["cnn_v2"]
    wav2vec = results["wav2vec2"]

    agreement = (
        cnn["prediction"] == wav2vec["prediction"]
    )

    cnn_info = get_score_strength(cnn)
    wav_info = get_score_strength(wav2vec)

    st.subheader("Overall Detection Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "CNN V2 Score Difference",
            f"{cnn_info['margin']:.2f} pp"
        )

    with col2:
        st.metric(
            "Wav2Vec2 Score Difference",
            f"{wav_info['margin']:.2f} pp"
        )

    st.divider()

    if not agreement:

        st.error(
            "MODEL DISAGREEMENT — INCONCLUSIVE RESULT"
        )

        st.warning(
            "The two detection models produced different predictions. "
            "VoiceShield cannot confidently determine the nature of "
            "this recording from these conflicting outputs."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**CNN V2**")
            st.write(cnn["prediction"])

        with col2:
            st.markdown("**Wav2Vec2**")
            st.write(wav2vec["prediction"])

        st.info(
            "Recommended action: Verify the speaker through a trusted "
            "phone number or another independent communication channel."
        )

    else:

        if cnn["prediction"] == "AI Generated":

            st.warning(
                "BOTH MODELS DETECT AN AI-GENERATED VOICE SIGNAL"
            )

            st.write(
                "Both models classify this recording as AI-generated. "
                "Treat it as suspicious and independently verify "
                "the source before trusting it."
            )

        else:

            st.success(
                "BOTH MODELS DETECT A HUMAN-LIKE VOICE SIGNAL"
            )

            st.write(
                "Both models classify this recording as human-like. "
                "However, agreement does not prove that the recording "
                "is authentic or has not been manipulated."
            )

    st.divider()

    st.markdown("**How to interpret this assessment**")

    st.markdown("""
    - **Agreement:** Both models produce the same classification.
    - **Disagreement:** The models produce different classifications.
    - **Strong model preference:** A larger difference between AI and
      Human scores.
    - **Close scores:** A smaller difference between AI and Human scores.
    """)

    st.caption(
        "Score strength is based on the difference between the two "
        "model scores. It is not calibrated confidence, accuracy, "
        "or a verified probability of authenticity."
    )


# ============================================================
# HISTORY ENTRY
# ============================================================

def create_history_entry(filename, results):

    cnn = results["cnn_v2"]
    wav = results["wav2vec2"]

    cnn_info = get_score_strength(cnn)
    wav_info = get_score_strength(wav)

    agreement = (
        cnn["prediction"] == wav["prediction"]
    )

    if not agreement:
        assessment = "Inconclusive - Disagreement"

    elif cnn["prediction"] == "AI Generated":
        assessment = "AI Signal - Both Models"

    else:
        assessment = "Human-like - Both Models"

    return {
        "Date & Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Audio File": filename,

        "CNN V2 Prediction": cnn["prediction"],

        "CNN V2 AI Score (%)": round(
            cnn["ai_score"] * 100, 2
        ),

        "CNN V2 Score Difference (%)": round(
            cnn_info["margin"], 2
        ),

        "Wav2Vec2 Prediction": wav["prediction"],

        "Wav2Vec2 AI Score (%)": round(
            wav["ai_score"] * 100, 2
        ),

        "Wav2Vec2 Score Difference (%)": round(
            wav_info["margin"], 2
        ),

        "Models Agree": "Yes" if agreement else "No",

        "Overall Assessment": assessment
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ VoiceShield AI")

    st.caption("Deep Learning Voice Security System")

    st.divider()

    st.subheader("Detection Engines")

    st.markdown("🧠 **CNN V2**")
    st.caption("Custom-trained deep-learning model.")

    st.markdown("🎧 **Wav2Vec2**")
    st.caption("Pretrained audio classification model.")

    st.divider()

    st.subheader("Project Capabilities")

    st.markdown("""
    - Audio analysis
    - Dual-model comparison
    - AI voice detection
    - Score strength analysis
    - Agreement and disagreement detection
    - Prediction history
    - CSV reports
    - Security recommendations
    """)

    st.divider()

    st.metric(
        "Completed Analyses",
        len(st.session_state.history)
    )

    st.caption("Educational research prototype")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🛡️ VoiceShield AI")

st.subheader(
    "AI-Powered Voice Clone Detection and Impersonation Prevention"
)

st.markdown(
    "Protecting digital communication through custom deep learning "
    "and pretrained AI models."
)

st.divider()

st.info(
    "VoiceShield compares two independent AI detection models. "
    "Their predictions are estimates, not proof that a voice is "
    "genuine or cloned. Always verify sensitive requests independently."
)


# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Audio Analysis",
    "🎙️ Microphone",
    "📊 History & Reports",
    "🛡️ Prevention"
])


# ============================================================
# TAB 1: AUDIO ANALYSIS
# ============================================================

with tab1:

    st.header("Audio Detection")

    st.write(
        "Upload a voice recording to analyze it using both "
        "your custom CNN V2 and pretrained Wav2Vec2 model."
    )

    audio_file = st.file_uploader(
        "Choose an audio recording",
        type=[
            "wav", "mp3", "ogg",
            "m4a", "flac", "webm"
        ],
        key="audio_upload"
    )

    if audio_file:

        st.success(
            f"Uploaded successfully: {audio_file.name}"
        )

        st.audio(audio_file)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "File Size",
                f"{audio_file.size / 1024:.2f} KB"
            )

        with col2:
            extension = audio_file.name.split(".")[-1].upper()

            st.metric("File Format", extension)

        if st.button(
            "🚀 Analyze with Both Models",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing audio with CNN V2 and Wav2Vec2..."
                ):

                    results = analyze_audio(
                        audio_file.getvalue()
                    )

                st.session_state.analysis_result = results

                st.session_state.analysis_filename = (
                    audio_file.name
                )

                entry = create_history_entry(
                    audio_file.name,
                    results
                )

                st.session_state.history.insert(0, entry)

                st.success(
                    "Both models completed their analysis!"
                )

            except Exception as error:

                st.error(
                    f"Audio analysis failed: {error}"
                )

    # Display latest analysis

    if st.session_state.analysis_result is not None:

        results = st.session_state.analysis_result

        st.divider()

        st.header("Dual-Model Detection Results")

        st.caption(
            f"Analyzed file: {st.session_state.analysis_filename}"
        )

        col1, col2 = st.columns(2)

        with col1:

            display_model_result(
                "🧠 CNN V2",
                "Custom-trained deep-learning model",
                results["cnn_v2"]
            )

        with col2:

            display_model_result(
                "🎧 Wav2Vec2",
                "Pretrained audio classification model",
                results["wav2vec2"]
            )

        st.divider()

        st.subheader("Model Score Comparison")

        comparison_df = pd.DataFrame({

            "Model": ["CNN V2", "Wav2Vec2"],

            "AI Generated": [
                results["cnn_v2"]["ai_score"] * 100,
                results["wav2vec2"]["ai_score"] * 100
            ],

            "Human Voice": [
                results["cnn_v2"]["human_score"] * 100,
                results["wav2vec2"]["human_score"] * 100
            ]

        })

        st.bar_chart(
            comparison_df.set_index("Model")
        )

        st.divider()

        display_assessment(results)

        st.divider()

        st.subheader("Technical Analysis Summary")

        summary_df = pd.DataFrame({

            "Parameter": [
                "CNN V2 Prediction",
                "Wav2Vec2 Prediction",
                "CNN V2 Score Strength",
                "Wav2Vec2 Score Strength",
                "Model Agreement"
            ],

            "Result": [
                results["cnn_v2"]["prediction"],
                results["wav2vec2"]["prediction"],
                get_score_strength(results["cnn_v2"])["strength"],
                get_score_strength(results["wav2vec2"])["strength"],
                "Yes" if (
                    results["cnn_v2"]["prediction"]
                    == results["wav2vec2"]["prediction"]
                ) else "No"
            ]

        })

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The displayed scores are model outputs, not calibrated "
            "probabilities. Score strength describes the separation "
            "between AI and Human scores, not model correctness."
        )


# ============================================================
# TAB 2: MICROPHONE
# ============================================================

with tab2:

    st.header("🎙️ Microphone Recording")

    st.write(
        "Record your voice using your browser microphone. "
        "Download the recording and upload it in Audio Analysis."
    )

    components.html("""
        <div style="
            background:linear-gradient(135deg,#10243e,#163b56);
            padding:22px;
            border:1px solid #315575;
            border-radius:14px;
            color:white;
            font-family:Arial;
        ">

            <h3 style="color:#e0f2fe;">
                🎙️ Voice Recorder
            </h3>

            <button id="start" style="
                padding:12px 20px;
                background:#0891b2;
                color:white;
                border:0;
                border-radius:8px;
                cursor:pointer;
                font-weight:bold;
            ">
                Start Recording
            </button>

            <button id="stop" style="
                padding:12px 20px;
                background:#dc2626;
                color:white;
                border:0;
                border-radius:8px;
                cursor:pointer;
                font-weight:bold;
                margin-left:8px;
            " disabled>
                Stop Recording
            </button>

            <p id="status" style="color:#bae6fd;">
                Microphone is not active.
            </p>

            <audio id="player" controls style="width:100%;"></audio>

            <p id="download"></p>

        </div>

        <script>

        let recorder;
        let chunks = [];
        let stream;

        const start = document.getElementById("start");
        const stop = document.getElementById("stop");
        const status = document.getElementById("status");
        const player = document.getElementById("player");
        const download = document.getElementById("download");

        start.onclick = async () => {

            try {

                stream = await navigator.mediaDevices.getUserMedia({
                    audio: true
                });

                let mimeType = "";

                if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
                    mimeType = "audio/webm;codecs=opus";
                }

                recorder = new MediaRecorder(
                    stream,
                    mimeType ? {mimeType: mimeType} : {}
                );

                chunks = [];

                recorder.ondataavailable = event => {

                    if (event.data.size > 0) {
                        chunks.push(event.data);
                    }

                };

                recorder.onstop = () => {

                    const blob = new Blob(chunks, {
                        type: recorder.mimeType
                    });

                    const url = URL.createObjectURL(blob);

                    player.src = url;

                    const link = document.createElement("a");

                    link.href = url;
                    link.download = "voice_recording.webm";
                    link.textContent = "⬇ Download Recording";

                    link.style.color = "#67e8f9";
                    link.style.fontSize = "16px";

                    download.innerHTML = "";
                    download.appendChild(link);

                    stream.getTracks().forEach(track => {
                        track.stop();
                    });

                    status.textContent =
                        "Recording completed. Download your audio.";

                };

                recorder.start();

                start.disabled = true;
                stop.disabled = false;

                status.textContent =
                    "🔴 Recording active. Speak now.";

            } catch (error) {

                status.textContent =
                    "Microphone access failed: " + error.message;

            }

        };

        stop.onclick = () => {

            if (recorder && recorder.state !== "inactive") {
                recorder.stop();
            }

            start.disabled = false;
            stop.disabled = true;

        };

        </script>
    """, height=310)

    st.info(
        "This feature records audio in your browser. "
        "It does not yet perform continuous real-time detection. "
        "Upload the downloaded recording for analysis."
    )


# ============================================================
# TAB 3: HISTORY AND REPORTS
# ============================================================

with tab3:

    st.header("📊 Analysis History")

    history = st.session_state.history

    if history:

        history_df = pd.DataFrame(history)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Analyses",
                len(history)
            )

        with col2:

            suspicious = sum(
                row["Overall Assessment"]
                == "AI Signal - Both Models"
                for row in history
            )

            st.metric(
                "Agreed AI Signals",
                suspicious
            )

        with col3:

            disagreements = sum(
                row["Models Agree"] == "No"
                for row in history
            )

            st.metric(
                "Model Disagreements",
                disagreements
            )

        st.divider()

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download Analysis Report (CSV)",
            data=csv_data,
            file_name="voiceshield_analysis_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "Clear Session History",
            use_container_width=True
        ):

            st.session_state.history = []
            st.session_state.analysis_result = None
            st.session_state.analysis_filename = None

            st.rerun()

    else:

        st.info(
            "No analysis history yet. Upload an audio file "
            "and run both models to create your first report."
        )


# ============================================================
# TAB 4: PREVENTION
# ============================================================

with tab4:

    st.header("🛡️ Voice Impersonation Prevention")

    st.warning(
        "Voice cloning can be used to impersonate family members, "
        "employees and other trusted individuals."
    )

    st.subheader("Security Recommendations")

    st.markdown("""
    **1. Verify the caller**

    Contact the person using a previously known phone number.

    **2. Use a family verification phrase**

    Agree on a private phrase with family members.

    **3. Protect sensitive information**

    Never share OTPs, passwords or banking credentials.

    **4. Verify financial requests**

    Confirm payment requests through a second communication channel.

    **5. Watch for urgency**

    Be cautious when someone pressures you to act immediately.

    **6. Do not rely on voice alone**

    A familiar-sounding voice is not sufficient proof of identity.
    """)

    st.divider()

    st.subheader("How VoiceShield AI Works")

    st.markdown("""
    1. **Audio Input:** The user uploads or records audio.
    2. **Preprocessing:** Audio is converted into mono 16 kHz format.
    3. **CNN V2:** A custom-trained CNN analyzes Mel-spectrogram features.
    4. **Wav2Vec2:** A pretrained audio model analyzes the recording.
    5. **Comparison:** Both model predictions and score differences are compared.
    6. **Assessment:** VoiceShield presents results and recommends verification.
    """)

    st.success(
        "AI can help identify suspicious audio, "
        "but independent verification remains essential."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("### 🛡️ VoiceShield AI")

st.caption(
    "AI-Powered Voice Clone Detection and Impersonation Prevention"
)

st.caption(
    "Educational Deep Learning Research Prototype"
)

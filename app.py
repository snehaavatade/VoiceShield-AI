
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
# REUSABLE FUNCTIONS
# ============================================================

def display_model_result(model_name, description, result):

    ai_score = result["ai_score"] * 100
    human_score = result["human_score"] * 100

    prediction = result["prediction"]

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


def display_assessment(results):

    cnn = results["cnn_v2"]
    wav2vec = results["wav2vec2"]

    agreement = results["agreement"]

    st.subheader("Overall Detection Assessment")

    if not agreement:

        st.warning(
            "⚠️ MODEL DISAGREEMENT\n\n"
            "The two models produced different predictions. "
            "Additional verification is recommended."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**CNN V2**")
            st.write(cnn["prediction"])

        with col2:
            st.write("**Wav2Vec2**")
            st.write(wav2vec["prediction"])

    elif cnn["prediction"] == "AI Generated":

        st.error(
            "⚠️ SUSPICIOUS AUDIO SIGNAL\n\n"
            "Both models classify this recording as AI-generated. "
            "Treat the recording cautiously and verify its source."
        )

    else:

        st.success(
            "✓ HUMAN-LIKE AUDIO SIGNAL\n\n"
            "Both models classify this recording as human-like. "
            "This does not prove that the recording is authentic."
        )

    st.caption(
        "Model agreement does not guarantee correctness. "
        "Confidence scores are not calibrated probabilities."
    )


def create_history_entry(filename, results):

    return {
        "Date & Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Audio File": filename,

        "CNN V2 Prediction": (
            results["cnn_v2"]["prediction"]
        ),

        "CNN V2 AI Score (%)": round(
            results["cnn_v2"]["ai_score"] * 100, 2
        ),

        "Wav2Vec2 Prediction": (
            results["wav2vec2"]["prediction"]
        ),

        "Wav2Vec2 AI Score (%)": round(
            results["wav2vec2"]["ai_score"] * 100, 2
        ),

        "Models Agree": (
            "Yes" if results["agreement"] else "No"
        )
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
    - Confidence comparison
    - Prediction history
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
    "VoiceShield compares two AI detection models. "
    "Results are estimates, not proof that a voice is genuine "
    "or cloned. Always verify sensitive requests independently."
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
            "wav",
            "mp3",
            "ogg",
            "m4a",
            "flac",
            "webm"
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

            st.metric(
                "File Format",
                extension
            )

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

                st.session_state.history.insert(
                    0,
                    entry
                )

                st.success(
                    "Both models completed their analysis!"
                )

            except Exception as error:

                st.error(
                    f"Audio analysis failed: {error}"
                )

    # Display the latest result
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

            "Model": [
                "CNN V2",
                "Wav2Vec2"
            ],

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

        st.caption(
            "These scores are model outputs, not calibrated probabilities. "
            "They should not be interpreted as a guaranteed measure "
            "of voice authenticity."
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
                row["CNN V2 Prediction"] == "AI Generated"
                or row["Wav2Vec2 Prediction"] == "AI Generated"
                for row in history
            )

            st.metric(
                "Analyses with AI Signal",
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
    5. **Comparison:** Both model predictions and scores are compared.
    6. **Assessment:** VoiceShield presents the results and recommends verification.
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

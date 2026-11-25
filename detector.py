
import io
import subprocess
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "voiceshield_cnn_v2.pth"

WAV2VEC_MODEL_ID = "Hemgg/Deepfake-audio-detection"

SAMPLE_RATE = 16000
AUDIO_LENGTH = 5
NUM_SAMPLES = SAMPLE_RATE * AUDIO_LENGTH

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CNN V2 ARCHITECTURE
# Must match train_cnn_v2.py exactly
# ============================================================

class VoiceShieldCNNV2(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD WAV2VEC2
# ============================================================

@lru_cache(maxsize=1)
def load_wav2vec_detector():

    classifier = pipeline(
        "audio-classification",
        model=WAV2VEC_MODEL_ID,
        device=0 if DEVICE.type == "cuda" else -1
    )

    return classifier


# ============================================================
# LOAD CUSTOM CNN V2
# ============================================================

@lru_cache(maxsize=1)
def load_cnn_v2():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"CNN V2 checkpoint not found: {MODEL_PATH}"
        )

    model = VoiceShieldCNNV2().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_audio(audio_bytes):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1"
    ]

    process = subprocess.run(
        command,
        input=audio_bytes,
        capture_output=True,
        check=False
    )

    if process.returncode != 0:

        raise ValueError(
            "Could not decode this audio file. "
            "Please try a WAV, MP3 or WEBM recording."
        )

    return process.stdout


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def prepare_audio(audio_bytes):

    converted_audio = convert_audio(audio_bytes)

    audio, sample_rate = librosa.load(
        io.BytesIO(converted_audio),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < SAMPLE_RATE:

        raise ValueError(
            "Audio is too short. Please provide at least 1 second."
        )

    if np.max(np.abs(audio)) < 0.001:

        raise ValueError(
            "The uploaded audio appears to be silent."
        )

    audio = torch.tensor(
        audio,
        dtype=torch.float32
    )

    audio = torch.nan_to_num(audio)

    # Same 5-second center crop/padding used during CNN training
    if len(audio) > NUM_SAMPLES:

        start = (len(audio) - NUM_SAMPLES) // 2

        audio = audio[
            start:start + NUM_SAMPLES
        ]

    elif len(audio) < NUM_SAMPLES:

        padding = NUM_SAMPLES - len(audio)

        audio = torch.nn.functional.pad(
            audio,
            (0, padding)
        )

    return audio


# ============================================================
# CREATE CNN MEL SPECTROGRAM
# ============================================================

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

db_transform = torchaudio.transforms.AmplitudeToDB(
    stype="power",
    top_db=80
)


def create_spectrogram(audio):

    mel = mel_transform(audio)

    mel = db_transform(mel)

    # Same normalization used during training
    mel = (
        mel - mel.mean()
    ) / (mel.std() + 1e-6)

    return mel.unsqueeze(0).unsqueeze(0)


# ============================================================
# WAV2VEC2 ANALYSIS
# ============================================================

def analyze_wav2vec(audio):

    classifier = load_wav2vec_detector()

    results = classifier({
        "raw": audio.numpy(),
        "sampling_rate": SAMPLE_RATE
    })

    ai_score = 0.0
    human_score = 0.0

    for item in results:

        label = str(item["label"]).lower()

        score = float(item["score"])

        if any(
            word in label
            for word in [
                "ai",
                "fake",
                "spoof",
                "synthetic",
                "generated"
            ]
        ):

            ai_score += score

        else:

            human_score += score

    total = ai_score + human_score

    if total <= 0:

        raise ValueError(
            "Wav2Vec2 did not return valid class scores."
        )

    ai_score /= total
    human_score /= total

    prediction = (
        "AI Generated"
        if ai_score >= human_score
        else "Human"
    )

    return {

        "ai_score": ai_score,

        "human_score": human_score,

        "prediction": prediction

    }


# ============================================================
# CUSTOM CNN V2 ANALYSIS
# ============================================================

def analyze_cnn_v2(audio):

    model = load_cnn_v2()

    spectrogram = create_spectrogram(audio).to(DEVICE)

    with torch.no_grad():

        outputs = model(spectrogram)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    human_score = float(probabilities[0].item())

    ai_score = float(probabilities[1].item())

    prediction = (
        "AI Generated"
        if ai_score >= human_score
        else "Human"
    )

    return {

        "ai_score": ai_score,

        "human_score": human_score,

        "prediction": prediction

    }


# ============================================================
# MAIN DUAL-MODEL ANALYSIS
# ============================================================

def analyze_audio(audio_bytes):

    # Convert and preprocess once
    audio = prepare_audio(audio_bytes)

    # Run both models
    cnn_results = analyze_cnn_v2(audio)

    wav2vec_results = analyze_wav2vec(audio)

    # Compare model decisions
    agreement = (
        cnn_results["prediction"]
        == wav2vec_results["prediction"]
    )

    return {

        "cnn_v2": cnn_results,

        "wav2vec2": wav2vec_results,

        "agreement": agreement,

        "final_status": (
            "Models agree"
            if agreement
            else "Models disagree"
        ),

        "audio_duration_seconds": (
            len(audio) / SAMPLE_RATE
        )

    }

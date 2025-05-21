import streamlit as st
import requests
import base64
import os
from io import BytesIO
from pydub import AudioSegment
from dotenv import load_dotenv
from audiorecorder import audiorecorder

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq models
WHISPER_MODEL = "distil-whisper-large-v3-en"
LLM_MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Session state
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

st.set_page_config(page_title="AudioPen Clone", layout="centered")
st.title("🎙️ AudioPen Clone using Streamlit + Groq")

# --- Record Audio ---
st.subheader("🎤 Record Audio")
recorded_audio = audiorecorder("Click to record", "Recording...")

# Store audio in session state
if recorded_audio:
    st.session_state.audio_data = recorded_audio

# --- If audio is available ---
if st.session_state.audio_data:
    # Playback
    wav_buffer = BytesIO()
    st.session_state.audio_data.export(wav_buffer, format="wav")
    st.audio(wav_buffer, format="audio/wav")

    # Buttons
    st.subheader("🎛️ Audio Controls")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Submit Audio"):
            mp3_io = BytesIO()
            st.session_state.audio_data.export(mp3_io, format="mp3")
            mp3_io.seek(0)

            # --- Transcription via Groq Whisper ---
            with st.spinner("📝 Transcribing..."):
                files = {
                    "file": ("audio.mp3", mp3_io, "audio/mpeg")
                }
                data = {
                    "model": WHISPER_MODEL,
                    "response_format": "text"
                }
                whisper_response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    data=data,
                    files=files
                )
                if whisper_response.status_code != 200:
                    st.error("❌ Transcription failed: " + whisper_response.text)
                    st.stop()

                st.session_state.transcript = whisper_response.text

            # --- Summarization via Groq LLM ---
            with st.spinner("🧠 Summarizing..."):
                summary_payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that summarizes speech text."},
                        {"role": "user", "content": f"Summarize the following:\n\n{st.session_state.transcript}"}
                    ]
                }
                llm_response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=HEADERS,
                    json=summary_payload
                )
                if llm_response.status_code != 200:
                    st.error("❌ Summary generation failed: " + llm_response.text)
                    st.stop()

                st.session_state.summary = llm_response.json()["choices"][0]["message"]["content"]

    with col2:
        if st.button("🗑️ Clear Audio"):
            st.session_state.audio_data = None
            st.session_state.transcript = ""
            st.session_state.summary = ""
            st.success("Audio and output cleared.")
            st.stop()

# --- Show Results ---
if st.session_state.transcript:
    st.subheader("🗒️ Transcription")
    st.text_area("Raw Text", st.session_state.transcript, height=150)

if st.session_state.summary:
    st.subheader("📌 Summary")
    st.text_area("Refined Summary", st.session_state.summary, height=150)

# --- No Audio Yet ---
if not st.session_state.audio_data:
    st.info("⬆️ Record audio using the button above.")

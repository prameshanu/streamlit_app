import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
import os

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frames):
        self.frames.extend(frames)
        return frames

def process_audio_to_text(audio_file_path):
    """Convert the recorded audio file to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Speech not recognized."
        except sr.RequestError:
            return "Error with the speech recognition service."

def main():
    st.title("Streamlit Voice to Text App")

    # Record audio
    st.header("Step 1: Record Your Audio")
    with st.expander("Click to Record"):
        webrtc_ctx = webrtc_streamer(
            key="voice-recording",
            audio_processor_factory=AudioProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        )

        if webrtc_ctx and webrtc_ctx.state.playing:
            st.info("Recording... Stop when you're done.")

    # Save the recorded audio
    if webrtc_ctx and not webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        audio_frames = webrtc_ctx.audio_processor.frames

        if audio_frames:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                AudioSegment(
                    data=b"".join(audio_frames),
                    sample_width=2,
                    frame_rate=16000,
                    channels=1,
                ).export(audio_file.name, format="wav")
                st.success("Audio recorded successfully.")
                audio_path = audio_file.name
        else:
            st.warning("No audio recorded yet. Please record and stop to save.")

    # Convert audio to text
    st.header("Step 2: Convert Audio to Text")
    if "audio_path" in locals():
        with st.spinner("Converting audio to text..."):
            transcript = process_audio_to_text(audio_path)
            st.success("Conversion Completed")
            st.text_area("Transcript", transcript)
            os.remove(audio_path)  # Clean up temporary audio file

if __name__ == "__main__":
    main()

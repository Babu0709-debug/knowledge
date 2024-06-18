import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, VideoProcessorBase, WebRtcMode
import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_text = None

    def recv(self, frame):
        audio_data = frame.to_ndarray()
        audio_source = sr.AudioData(audio_data.tobytes(), frame.sample_rate, frame.sample_width)
        
        try:
            self.audio_text = self.recognizer.recognize_google(audio_source)
            st.session_state.audio_text = self.audio_text
        except sr.UnknownValueError:
            st.session_state.audio_text = "Could not understand audio"
        except sr.RequestError as e:
            st.session_state.audio_text = f"Could not request results; {e}"
        return frame

def main():
    st.title("Webcam and Audio Capture with Streamlit")

    if 'audio_text' not in st.session_state:
        st.session_state.audio_text = ""

    # Start the webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": True, "audio": True}
    )

    # Display the recognized text after stopping the stream
    if st.button("Stop Listening"):
        webrtc_ctx.stop()
        if st.session_state.audio_text:
            st.write(f"Recognized Text: {st.session_state.audio_text}")
        else:
            st.write("No audio captured or unable to recognize speech")

if __name__ == "__main__":
    main()

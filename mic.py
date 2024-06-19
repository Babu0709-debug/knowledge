import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, VideoProcessorBase, WebRtcMode
import speech_recognition as sr
import numpy as np
import av

# Initialize the recognizer
recognizer = sr.Recognizer()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_text = None
        self.is_recording = False

    def recv(self, frame):
        if self.is_recording:
            audio_frame = frame.to_ndarray()
            if audio_frame.ndim == 1:
                audio_frame = np.expand_dims(audio_frame, axis=1)
            audio_data = av.AudioFrame.from_ndarray(audio_frame, layout="mono")
            audio_source = sr.AudioData(audio_data.to_bytes(), frame.sample_rate, frame.sample_width)
            
            try:
                self.audio_text = self.recognizer.recognize_google(audio_source)
                st.session_state.audio_text = self.audio_text
            except sr.UnknownValueError:
                st.session_state.audio_text = "Could not understand audio"
            except sr.RequestError as e:
                st.session_state.audio_text = f"Could not request results; {e}"
        return frame

    def start_recording(self):
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_image()
        return frame

def main():
    st.title("Webcam and Audio Capture with Streamlit")

    if 'audio_text' not in st.session_state:
        st.session_state.audio_text = ""

    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None

    # Initialize the webrtc context
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": True}
    )

    if webrtc_ctx.video_processor:
        capture_image_button = st.button("Capture Image")
        if capture_image_button:
            st.session_state.captured_image = np.array(webrtc_ctx.video_processor.frame)
            if st.session_state.captured_image is not None:
                st.image(st.session_state.captured_image, caption="Captured Image")
                # Save image to local storage
                with open("captured_image.png", "wb") as f:
                    f.write(cv2.imencode('.png', cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_RGB2BGR))[1].tobytes())

    if webrtc_ctx.audio_processor:
        start_recording_button = st.button("Start Recording")
        stop_recording_button = st.button("Stop Listening")

        if start_recording_button:
            webrtc_ctx.audio_processor.start_recording()

        if stop_recording_button:
            webrtc_ctx.audio_processor.stop_recording()
            if st.session_state.audio_text:
                st.write(f"Recognized Text: {st.session_state.audio_text}")
            else:
                st.write("No audio captured or unable to recognize speech")

if __name__ == "__main__":
    main()

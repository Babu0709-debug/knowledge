import streamlit as st
import sounddevice as sd  # for microphone recording
import soundfile as sf  # for audio file handling
import whisper  # OpenAI Whisper library

def listen_and_recognize():

  # Microphone recording parameters
  duration = 5  # Recording duration in seconds
  fs = 16000  # Sampling rate

  # Record audio from microphone
  try:
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Done recording!")
  except Exception as e:
    st.error(f"Error recording audio: {e}")
    return None

  # Convert recording to audio buffer
  audio_buffer = myrecording.copy()

  # Use Whisper model for speech recognition
  result = model.transcribe(audio_buffer)

  st.write("You said: " + result["text"])
  return result["text"]

def main():
  st.title("Speech Recognition Chatbox (OpenAI Whisper)")

  # Load Whisper model (Optional)
  model = whisper.load_model("base")  # Options: "base", "medium", "large"

  # ... (rest of your Streamlit UI logic)

  if st.button("Recognize Speech from Microphone"):
    query = listen_and_recognize()

    # ... (handle recognized text, display chat messages, etc.)

if __name__ == "__main__":
  main()

import streamlit as st
import speech_recognition as sr

def listen_and_recognize():
    """Listens for user speech and returns recognized text."""

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        st.write("You said: " + query)
        return query
    except sr.UnknownValueError:
        st.error("Sorry, could not understand audio. Please try again.")
        return None

def main():
    """Main function to display UI and handle speech recognition."""

    st.title("Speech Recognition Chatbox")

    # Initialize session state variables (optional)
    if 'listening' not in st.session_state:
        st.session_state.listening = False

    # Button for starting/stopping listening
    if st.button("Start Listening"):
        st.session_state.listening = True
    if st.button("Stop Listening") and 'listening' in st.session_state:
        st.session_state.listening = False

    # Handle listening state
    if st.session_state.listening:
        st.session_state.query = listen_and_recognize()

    # Display chat messages (optional)
    # ... (implement chat message display logic if desired)

if __name__ == "__main__":
    main()

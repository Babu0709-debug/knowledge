import streamlit as st
import speech_recognition as sr
import pyaudio

# Initialize the recognizer
recognizer = sr.Recognizer()

def listen_and_recognize():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write("You said: " + query)
        return query
    except sr.UnknownValueError:
        st.write("Sorry, could not understand audio. Please try again.")
        return None

def main():
    st.title("Speech Recognition Chatbox")

    # Initialize session state variables
    if 'listening' not in st.session_state:
        st.session_state.listening = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Define start and stop actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Listening"):
            st.session_state.listening = True
    with col2:
        if st.button("Stop Listening"):
            st.session_state.listening = False
            if 'query' in st.session_state:
                st.session_state.messages.append(st.session_state.query)
                del st.session_state.query

    # Handle listening state
    if st.session_state.listening:
        st.session_state.query = listen_and_recognize()

    # Display chat messages
    st.subheader("Chatbox")
    for message in st.session_state.messages:
        st.write(message)

if __name__ == "__main__":
        main()

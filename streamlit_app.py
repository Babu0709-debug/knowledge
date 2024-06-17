import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from pydub import AudioSegment
from pydub.playback import play
import io

# Load environment variables
load_dotenv()

# Dictionary to store the extracted dataframes
data = {}

# Function to recognize speech from audio data
def recognize_speech(audio_data):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Error in the request to speech recognition service"

def main():
    st.set_page_config(page_title="PandasAI", page_icon="🐼")
    st.title("Chat with Your Data using PandasAI:🐼")

    # Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        # Activating Demo Data
        st.text("Data Setup: 📝")
        file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
        st.markdown(":green[*Please ensure the first row has the column names.*]")

        # Selecting LLM to use
        llm_type = st.selectbox(
            "Please select LLM",
            ('BambooLLM', 'gemini-pro'), index=0
        )

        # Adding users API Key
        user_api_key = st.text_input('Please add your API key', placeholder='Paste your API key here', type='password')

    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df])

        llm = get_LLM(llm_type, user_api_key)

        if llm:
            # Instantiating PandasAI agent
            analyst = get_agent(data, llm)

            # Starting the chat with the PandasAI agent
            chat_window(analyst)

    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")

# Function to get LLM
def get_LLM(llm_type, user_api_key):
    # Creating LLM object based on the llm type selected:
    try:
        if llm_type == 'BambooLLM':
            if user_api_key:
                os.environ["PANDASAI_API_KEY"] = user_api_key
            else:
                # If no API key provided, try to get it from environment variables
                os.environ["PANDASAI_API_KEY"] = os.getenv('PANDASAI_API_KEY')
            llm = BambooLLM()

        elif llm_type == 'gemini-pro':
            if user_api_key:
                genai.configure(api_key=user_api_key)
            else:
                # Configure the API key
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=user_api_key)

        return llm
    except Exception as e:
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")

# Function for chat window
def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("Explore your data with PandasAI?🧐")

    # Initializing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the message history on re-run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Printing the questions
            if 'question' in message:
                st.markdown(message["question"])
            # Printing the code generated and the evaluated code
            elif 'response' in message:
                # Getting the response
                st.write(message['response'])
            # Retrieving error messages
            elif 'error' in message:
                st.text(message['error'])
    
    # WebRTC for speech-to-text
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
        ),
    )

    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            audio = audio_frames[-1].to_ndarray()
            audio = AudioSegment(
                data=audio.tobytes(),
                sample_width=audio.dtype.itemsize,
                frame_rate=audio_frames[-1].rate,
                channels=1
            )

            # Convert audio to wav format for recognition
            audio_wav = io.BytesIO()
            audio.export(audio_wav, format="wav")
            audio_wav.seek(0)
            audio_data = sr.AudioFile(audio_wav)
            recognizer = sr.Recognizer()
            with audio_data as source:
                audio_content = recognizer.record(source)
                user_question = recognize_speech(audio_content)
                st.write(f"Recognized Speech: {user_question}")

    # Getting the questions from the users
    user_question = st.chat_input("What are you curious about? ")

    if user_question:
        # Displaying the user question in the chat message
        with st.chat_message("user"):
            st.markdown(user_question)
        # Adding user question to chat history
        st.session_state.messages.append({"role": "user", "question": user_question})

        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "response": response})
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []

    # Button for clearing history
    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def get_agent(data, llm):
    """
    The function creates an agent on the dataframes extracted from the uploaded files
    Args: 
        data: A Dictionary with the dataframes extracted from the uploaded data
        llm: llm object based on the ll type selected
    Output: PandasAI Agent
    """
    agent = Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})

    return agent

def extract_dataframes(raw_file):
    """
    This function extracts dataframes from the uploaded file/files
    Args: 
        raw_file: Upload_File object
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output: 
        dfs: a dictionary with the dataframes
    """
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df

    elif (raw_file.name.split('.')[1] == 'xlsx') or (raw_file.name.split('.')[1] == 'xls'):
        # Read the Excel file
        xls = pd.ExcelFile(raw_file)

        # Iterate through each sheet in the Excel file and store them into dataframes
        dfs = {}
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)

    # Return the dataframes
    return dfs

if __name__ == "__main__":
    main()

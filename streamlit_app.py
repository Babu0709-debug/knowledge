import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
import openai
from meta_ai_api import MetaAI
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from streamlit_mic_recorder import mic_recorder, speech_to_text
from pydub import AudioSegment
import av
import io
import json

# Load environment variables
load_dotenv()

# Dictionary to store the extracted dataframes
data = {}

def main():
    st.set_page_config(page_title="FP&A", page_icon="🤖")
    st.title("Talk With Babu's Data")

    # Side Menu Bar
    with st.sidebar:
        st.title("FP&A Analysis")
        st.text("Data Setup: 📝")
        file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])

        st.markdown(":green[*Please ensure the first row has the column names.*]")

        # Selecting LLM to use
        llm_type = st.selectbox(
            "Please select LLM",
            ('BambooLLM', 'gemini-pro', 'meta-ai', 'openai-gpt'), index=0
        )

        # Adding users API Key
        user_api_key = st.text_input('Please commit', placeholder='Paste your API key here', type='password')

    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!",
                          tuple(data.keys()), index=0
                          )
        st.dataframe(data[df])

        llm = get_LLM(llm_type, user_api_key)

        if llm:
            # Instantiating PandasAI agent if not using MetaAI
            if llm_type != 'meta-ai':
                analyst = get_agent(data, llm)
            else:
                analyst = llm

            # Starting the chat with the PandasAI agent
            chat_window(analyst, llm_type)

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

        elif llm_type == 'meta-ai':
            llm = MetaAI()  # Initialize MetaAI directly without API key

        elif llm_type == 'openai-gpt':
            if user_api_key:
                openai.api_key = user_api_key
            else:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            llm = openai.Completion.create(engine="davinci", prompt="", max_tokens=100)

        return llm
    except Exception as e:
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")

# Function for chat window
def chat_window(analyst, llm_type):
    with st.chat_message("assistant"):
        st.text("Explore Babu's Data")

    # Initializing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the message history on re-run
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message['response'])
            elif 'error' in message:
                st.text(message['error'])

    # Getting the questions from the users
    user_question = st.text_input("Enter your question here", value=speech_to_text(language='en'))

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "question": user_question})

        try:
            with st.spinner("Analyzing..."):
                if llm_type == 'meta-ai':
                    response = analyst.prompt(message=user_question)
                    try:
                        response_json = json.loads(response)
                        st.write(response_json)
                        st.session_state.messages.append({"role": "assistant", "response": response_json})
                    except json.JSONDecodeError:
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "response": response})
                else:
                    response = analyst.chat(user_question)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "response": response})

        except Exception as e:
            st.error("⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question or try again later.")
            st.session_state.messages.append({"role": "assistant", "error": str(e)})

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
    elif raw_file.name.split('.')[1] in ['xlsx', 'xls']:
        # Read the Excel file
        xls = pd.ExcelFile(raw_file)
        # Iterate through each sheet in the Excel file and store them into dataframes
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)

    # Return the dataframes
    return dfs

if __name__ == "__main__":
    main()

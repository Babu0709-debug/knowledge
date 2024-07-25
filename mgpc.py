import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from streamlit_mic_recorder import mic_recorder, speech_to_text
from pydub import AudioSegment
import av
import io

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
        llm_type = st.selectbox("Please select LLM", ('BambooLLM', 'gemini-pro', 'meta-ai'), index=0)

        # Adding users API Key
        user_api_key = st.text_input('Please commit', placeholder='Paste your API key here', type='password')

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
    # Creating LLM object based on the llm type selected
    try:
        if llm_type == 'BambooLLM':
            if user_api_key:
                os.environ["PANDASAI_API_KEY"] = user_api_key
            else:
                os.environ["PANDASAI_API_KEY"] = os.getenv('PANDASAI_API_KEY')

            llm = BambooLLM()

        elif llm_type == 'gemini-pro':
            if user_api_key:
                genai.configure(api_key=user_api_key)
            else:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=user_api_key)

        elif llm_type == 'meta-ai':
            from langchain_meta_ai import ChatMetaAI
            llm = ChatMetaAI()  # Meta AI does not require an API key

        return llm
    except Exception as e:
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")

# Function for chat window
def chat_window(analyst):
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

    user_question = st.text_input("Enter your question here", value=speech_to_text(language='en'))

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "question": user_question})

        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                formatted_response = format_response(response)
                st.write(formatted_response)
                st.session_state.messages.append({"role": "assistant", "response": formatted_response})
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"

    # Function to clear history
    def clear_chat_history():
        st.session_state.messages = []

    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def get_agent(data, llm):
    agent = Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    return agent

def extract_dataframes(raw_file):
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df
    elif (raw_file.name.split('.')[1] == 'xlsx') or (raw_file.name.split('.')[1] == 'xls'):
        xls = pd.ExcelFile(raw_file)
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)
    return dfs

def format_response(response):
    # Example formatted response for demonstration purposes
    formatted_response = {
        "message": response,
        "sources": [
            {"link": "https://www.wolframalpha.com/input?i=San+Francisco+weather+today+and+date", "title": "WolframAlpha"},
            {"link": "https://www.timeanddate.com/weather/usa/san-francisco", "title": "Weather for San Francisco, California, USA - timeanddate.com"},
            {"link": "https://www.accuweather.com/en/us/san-francisco/94103/weather-today/347629", "title": "Weather Today for San Francisco, CA | AccuWeather"},
            {"link": "https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629", "title": "San Francisco, CA Weather Forecast | AccuWeather"},
            {"link": "https://forecast.weather.gov/zipcity.php?inputstring=San%20francisco%2CCA", "title": "National Weather Service"},
            {"link": "https://www.wunderground.com/weather/us/ca/san-francisco", "title": "San Francisco, CA Weather Conditions | Weather Underground"}
        ]
    }
    return formatted_response

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
from meta_ai_api import MetaAI
import os
import openai

# Load environment variables
load_dotenv()

# Setting the API key directly for demonstration purposes
PANDASAI_API_KEY = "$2a$10$CBHk/Ecf9SmSQMpbjqX2Zed/9ITezJijO/k2iDnrTYAwB8DyZWbUi"
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

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
        llm_type = st.selectbox("Please select LLM", ('BambooLLM', 'gemini-pro', 'meta-ai', 'openai'), index=0)

        # Conditionally showing/hiding API Key input box
        user_api_key = None
        if llm_type != 'meta-ai':
            user_api_key = st.text_input('Please commit', placeholder='Paste your API key here', type='password')

    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df])

        llm = get_LLM(llm_type, user_api_key)

        if llm:
            if llm_type != 'meta-ai':
                # Instantiating PandasAI agent if not using MetaAI
                analyst = get_agent(data, llm)
                chat_window(analyst)
            else:
                # Handle MetaAI directly for general conversations
                meta_ai_chat_window()
    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")

def get_LLM(llm_type, user_api_key):
    try:
        if llm_type == 'BambooLLM':
            if user_api_key:
                os.environ["PANDASAI_API_KEY"] = user_api_key
            else:
                os.environ["PANDASAI_API_KEY"] = PANDASAI_API_KEY  # Set the key directly

            llm = BambooLLM()

        elif llm_type == 'gemini-pro':
            if user_api_key:
                genai.configure(api_key=user_api_key)
            else:
                genai.configure(api_key=GOOGLE_API_KEY)  # Set the key directly

            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=user_api_key)

        elif llm_type == 'openai':
            if user_api_key:
                openai.api_key = user_api_key
            else:
                openai.api_key = OPENAI_API_KEY  # Set the key directly

            llm = openai

        elif llm_type == 'meta-ai':
            llm = MetaAI()  # Meta AI does not require an API key

        return llm
    except Exception as e:
        st.error(f"Error in getting LLM: {e}")

def get_agent(data, llm):
    """
    The function creates an agent on the dataframes extracted from the uploaded files
    Args: 
        data: A Dictionary with the dataframes extracted from the uploaded data
        llm:  LLM object based on the llm type selected
    Output: PandasAI Agent or None
    """
    if llm:
        return Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    return None

def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("Explore Babu's Data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                st.write(message['response'])
            elif 'error' in message:
                st.text(message['error'])

    user_question = st.text_input("Enter your question here")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "question": user_question})

        try:
            with st.spinner("Analyzing..."):
                if analyst:
                    # Handle non-MetaAI LLM
                    response = analyst.chat(user_question)
                    st.write(response)  # Use st.write instead of print
                    st.session_state.messages.append({"role": "assistant", "response": response})
                elif isinstance(analyst, openai):
                    # Handle OpenAI LLM
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=user_question,
                        max_tokens=150
                    )
                    formatted_response = response.choices[0].text.strip()
                    st.write(formatted_response)  # Use st.write instead of print
                    st.session_state.messages.append({"role": "assistant", "response": formatted_response})
        except Exception as e:
            st.write(f"⚠️ Sorry, Couldn't generate the answer! Please try rephrasing your question. Error: {e}")

    def clear_chat_history():
        st.session_state.messages = []

    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def meta_ai_chat_window():
    st.title("Casual chat with Babu")

    user_input = st.text_input("Type your message:")

    if st.button("Submit Your Request"):
        ai = MetaAI()
        try:
            response = ai.prompt(message=user_input)
            print(response)  # Use st.write instead of print
            if response and response != "":
                st.write(response.get('message', 'No response from Babu. Try again!'))
            else:
                st.write("No response from Babu. Try again!")
        except Exception as e:
            st.error(f"Check your query: {e}")

def extract_dataframes(raw_file):
    """
    This function extracts dataframes from the uploaded file/files
    Args: 
        raw_file: Upload_File object
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output: 
        dfs:  a dictionary with the dataframes
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
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)

    # return the dataframes
    return dfs

if __name__ == "__main__":
    main()

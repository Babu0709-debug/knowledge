import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from meta_ai_api import MetaAI
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.llm import BambooLLM, OpenAI
from dotenv import load_dotenv
import os

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
            ('MetaAI', 'OpenAI', 'BambooLLM', 'gemini-pro'), index=0
        )
        
        # Adding user's API Key
        user_api_key = st.text_input('Please commit', placeholder='Paste your API key here', type='password')
    
    if file_upload is not None:
        data = extract_dataframes(file_upload)
        df_name = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df_name])
        
        llm = get_LLM(llm_type, user_api_key)
        
        if llm:
            # Instantiating PandasAI agent
            try:
                analyst = get_agent(data, llm)
                # Starting the chat with the PandasAI agent
                chat_window(analyst)
            except Exception as e:
                st.error(f"Failed to initialize agent: {e}")
    
    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")

# Function to get LLM
def get_LLM(llm_type, user_api_key):
    try:
        if llm_type == 'MetaAI':
            return MetaAI()  # Assuming MetaAI is an instance of the LLM class or compatible
        elif llm_type == 'OpenAI':
            return OpenAI(api_token=user_api_key)
        elif llm_type == 'BambooLLM':
            if user_api_key:
                os.environ["PANDASAI_API_KEY"] = user_api_key
            else:
                os.environ["PANDASAI_API_KEY"] = os.getenv('PANDASAI_API_KEY')
            return BambooLLM()
        elif llm_type == 'gemini-pro':
            if user_api_key:
                genai.configure(api_key=user_api_key)
            else:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=user_api_key)
    except Exception as e:
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")
        return None

# Function for chat window
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
                response = analyst.chat(user_question)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "response": response})
        
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"

    def clear_chat_history():
        st.session_state.messages = []
    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def get_agent(data, llm):
    try:
        # Check if llm is an instance of the expected LLM class
        if not isinstance(llm, (BambooLLM, OpenAI, MetaAI, ChatGoogleGenerativeAI)):
            raise ValueError(f"llm is not an instance of the expected type: {type(llm)}")
        
        config = {"llm": llm, "verbose": True, "response_parser": StreamlitResponse}
        st.write("Configuring agent with the following settings:", config)
        agent = Agent(list(data.values()), config=config)
        return agent
    except Exception as e:
        st.error(f"Error in agent configuration: {e}")
        raise e

def extract_dataframes(raw_file):
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df
    elif raw_file.name.split('.')[1] in ['xlsx', 'xls']:
        xls = pd.ExcelFile(raw_file)
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)
    return dfs

if __name__ == "__main__":
    main()

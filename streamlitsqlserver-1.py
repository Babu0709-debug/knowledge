import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
from meta_ai_api import MetaAI
import openai
import os
import tiktoken
import pyodbc
import mysql.connector
import warnings
from sqlalchemy import create_engine
st.set_page_config(page_title="Data Analysis with LLMs", page_icon="📊")
drivers = pyodbc.drivers()
st.write("Available ODBC Drivers:", drivers)
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Dictionary to store the extracted dataframes
data = {}

def count_tokens(string: str) -> int:
    encoding_name = "p50k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_openai_response(input_text, openai_api_key):
    try:
        openai.api_key = openai_api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=input_text,
            max_tokens=150
        )
        num_tokens = count_tokens(input_text)
        st.info(f"Input contains {num_tokens} tokens.")
        st.info(response.choices[0].text.strip())
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.title("Data Analysis with LLMs")

    # Side Menu Bar
    with st.sidebar:
        st.title("FP&A Analysis")
        st.text("Data Setup: 📝")

        # Selection Dropdown for Data Source
        data_source = st.selectbox("Select Data Source", ["SQL Server", "MySQL", "Excel"], index=0)

        if data_source in ["SQL Server", "MySQL"]:
            st.subheader(f"{data_source} Connection")

            # Choose the SQL Server or MySQL
            if data_source == "SQL Server":
                server_name = st.text_input("Server Name", "")
                database_name = st.text_input("Database Name", "")
                query = st.text_area("SQL Query", "SELECT 1")

            elif data_source == "MySQL":
                host = st.text_input("Host", "")
                user = st.text_input("User", "")
                password = st.text_input("Password", type="password")
                database_name = st.text_input("Database Name", "")
                query = st.text_area("SQL Query", "SELECT 1")

        else:
            file_upload = st.file_uploader("Upload your Data", accept_multiple_files=False, type=['csv', 'xls', 'xlsx'])
            st.markdown(":green[*Please ensure the first row has the column names.*]")

        # Selecting LLM to use
        llm_type = st.selectbox("Please select LLM", ('BambooLLM', 'gemini-pro', 'meta-ai', 'openai'), index=0)

        # Adding user's API Key
        if llm_type != 'meta-ai':
            user_api_key = st.text_input('Please commit', placeholder='Paste your API key here', type='password')

    # Handling SQL Server Connection
    if data_source == "SQL Server" and server_name and database_name and query:
        try:
            conn_str = (
                'DRIVER={SQL Server};'
                f'SERVER={server_name};'
                f'DATABASE={database_name};'
                'Trusted_Connection=yes;'
                'Connection Timeout=30;'
                'Encrypt=yes;'
            )
            conn = pyodbc.connect(conn_str)
            df = pd.read_sql_query(query, conn)
            data['SQL_Query_Result'] = df
            st.success("Successfully fetched data from SQL Server")
            st.dataframe(df)

            handle_llm_selection(llm_type, user_api_key, df)

        except Exception as e:
            st.error(f"Failed to connect to SQL Server: {e}")

    # Handling MySQL Connection
    elif data_source == "MySQL" and host and user and password and database_name and query:
        try:
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database_name
            )
            df = pd.read_sql_query(query, conn)
            data['MySQL_Query_Result'] = df
            st.success("Successfully fetched data from MySQL")
            st.dataframe(df)

            handle_llm_selection(llm_type, user_api_key, df)

        except Exception as e:
            st.error(f"Failed to connect to MySQL: {e}")

    elif data_source == "Excel" and file_upload is not None:
        data.update(extract_dataframes(file_upload))
        df = st.selectbox("Here's your uploaded data!", tuple(data.keys()), index=0)
        st.dataframe(data[df])

        handle_llm_selection(llm_type, user_api_key, data[df])

    else:
        st.warning("Please provide the required inputs.")

def handle_llm_selection(llm_type, user_api_key, df):
    llm = get_LLM(llm_type, user_api_key if llm_type != 'meta-ai' else None)

    if llm:
        if llm_type != 'meta-ai' and llm_type != 'openai':
            # Instantiating PandasAI agent if not using MetaAI or OpenAI
            analyst = get_agent(data, llm)
            chat_window(analyst)
        elif llm_type == 'meta-ai':
            # Handle MetaAI directly
            chat_window(None, llm)
        elif llm_type == 'openai':
            openai_chat_window(df, user_api_key)

def get_LLM(llm_type, user_api_key):
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
            llm = MetaAI()  # Meta AI does not require an API key

        elif llm_type == 'openai':
            llm = "openai"

        return llm
    except Exception as e:
        st.error(f"Error in getting LLM: {e}")

def get_agent(data, llm):
    if llm:
        return Agent(list(data.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})
    return None

def chat_window(analyst, llm=None):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            elif 'response' in message:
                if message["role"] == "assistant" and llm == "meta-ai":
                    st.markdown(f'<div style="color:white;">{message["response"]}</div>', unsafe_allow_html=True)
                else:
                    st.write(message["response"])
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
                    formatted_response = response
                elif llm:
                    # Handle MetaAI LLM
                    response = llm.prompt(message=user_question, stream=False)
                    formatted_response = format_meta_ai_response(response)
                if llm and llm == "meta-ai":
                    st.markdown(f'<div style="color:white;">{formatted_response}</div>', unsafe_allow_html=True)
                else:
                    st.write(formatted_response)
                st.session_state.messages.append({"role": "assistant", "response": formatted_response})
        except Exception as e:
            st.write(f"⚠️ Sorry, Couldn't generate the answer! Please try rephrasing your question. Error: {e}")

    def clear_chat_history():
        st.session_state.messages = []

    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)

def format_meta_ai_response(response):
    return response["message"].replace('\n', '<br>')

def extract_dataframes(uploaded_file):
    data = {}
    try:
        if uploaded_file.name.split(".")[-1] in ["xls", "xlsx"]:
            xl = pd.ExcelFile(uploaded_file)
            for sheet in xl.sheet_names:
                data[sheet] = xl.parse(sheet)
        elif uploaded_file.name.split(".")[-1] == "csv":
            data["CSV Data"] = pd.read_csv(uploaded_file)
        st.success("Data successfully extracted!")
        return data
    except Exception as e:
        st.error(f"Could not read file. Error: {str(e)}")
        return {}

if __name__ == '__main__':
    main()

# llm_app.py
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
from meta_ai_api import MetaAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient
import openai
import os
import pyodbc
import warnings
import tiktoken

warnings.filterwarnings('ignore')
load_dotenv()
data = {}  # Global dict for storing uploaded or queried data

def count_tokens(text):
    enc = tiktoken.get_encoding("p50k_base")
    return len(enc.encode(text))

def extract_dataframes(raw_file):
    dfs = {}
    if raw_file.name.endswith('.csv'):
        dfs[raw_file.name] = pd.read_csv(raw_file)
    else:
        xls = pd.ExcelFile(raw_file)
        for sheet in xls.sheet_names:
            dfs[sheet] = pd.read_excel(xls, sheet_name=sheet)
    return dfs

def get_llm(llm_type, user_api_key=None):
    if llm_type == 'BambooLLM':
        os.environ['PANDASAI_API_KEY'] = user_api_key or os.getenv('PANDASAI_API_KEY')
        return BambooLLM()

    elif llm_type == 'gemini-pro':
        genai.configure(api_key=user_api_key or os.getenv('GOOGLE_API_KEY'))
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    elif llm_type == 'gemma-free':
        genai.configure(api_key="AIzaSyDHr0wdLhKyXvQCj4HvmpDjriSNoBcSApU")  # Public/free demo key
        return genai.GenerativeModel("gemma-3-27b-it")

    elif llm_type == 'meta-ai':
        return MetaAI()

    elif llm_type == 'huggingface':
        return InferenceClient(model="google/flan-t5-xl")  # Or any open-access HF model

    elif llm_type == 'openai':
        openai.api_key = user_api_key or os.getenv("OPENAI_API_KEY")
        return "openai"

    return None

def get_agent(dataframes, llm):
    return Agent(list(dataframes.values()), config={"llm": llm, "verbose": True, "response_parser": StreamlitResponse})

def chat_window(analyst=None, llm=None):
    st.markdown("### 🤖 Chat with your Data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the data...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    if analyst:
                        result = analyst.chat(user_input)
                    elif llm == "openai":
                        prompt = f"Answer this question using your general knowledge:\n{user_input}"
                        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=200)
                        result = response.choices[0].text.strip()
                    elif isinstance(llm, genai.GenerativeModel):
                        result = llm.generate_content(user_input).text
                    elif isinstance(llm, InferenceClient):
                        result = llm.text_generation(user_input, max_new_tokens=200)
                    elif isinstance(llm, MetaAI):
                        result = llm.prompt(message=user_input)['message']
                    else:
                        result = "LLM not properly configured."

                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    st.error(f"Error: {e}")


def main():
    st.set_page_config(page_title="LLM Data Analysis", layout="wide")
    st.title("📊 Multimodal LLM Data Analyst")

    with st.sidebar:
        st.header("🛠️ Configuration")
        source = st.selectbox("Data Source", ["SQL", "Excel"])

        if source == "SQL":
            username = st.text_input("SQL Username", "celonis_dbread")
            password = st.text_input("Password", type="password")
            server = st.text_input("Server", "10.232.70.46")
            database = st.text_input("Database", "Ods_live")
            query = st.text_area("SQL Query", "SELECT TOP 10 * FROM EMOS.Sales_Invoiced")
        else:
            file = st.file_uploader("Upload Excel/CSV", type=['csv', 'xls', 'xlsx'])

        llm_type = st.selectbox("Choose LLM", ['BambooLLM', 'gemini-pro', 'gemma-free', 'meta-ai', 'huggingface', 'openai'])

        api_key = None
        if llm_type in ['BambooLLM', 'gemini-pro', 'openai']:
            api_key = st.text_input("API Key", type="password")

    if source == "SQL" and username and password and server and database and query:
        try:
            conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
            conn = pyodbc.connect(conn_str)
            df = pd.read_sql_query(query, conn)
            data['SQL'] = df
            st.success("✅ Data loaded from SQL")
            st.dataframe(df)
            llm = get_llm(llm_type, api_key)
            if llm_type in ['openai', 'meta-ai', 'gemma-free', 'huggingface']:
                chat_window(None, llm)
            else:
                agent = get_agent(data, llm)
                chat_window(agent)
        except Exception as e:
            st.error(f"❌ SQL Error: {e}")

    elif source == "Excel" and file is not None:
        try:
            data.update(extract_dataframes(file))
            df_name = st.selectbox("Select Sheet", list(data.keys()))
            st.dataframe(data[df_name])
            llm = get_llm(llm_type, api_key)
            if llm_type in ['openai', 'meta-ai', 'gemma-free', 'huggingface']:
                chat_window(None, llm)
            else:
                agent = get_agent(data, llm)
                chat_window(agent)
        except Exception as e:
            st.error(f"❌ File processing error: {e}")
    else:
        st.warning("⚠️ Provide data to begin.")

if __name__ == '__main__':
    main()

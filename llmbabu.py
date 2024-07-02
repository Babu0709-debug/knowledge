import streamlit as st
from langchain.llms import OpenAI
import tiktoken

st.title('GeekBot')

openai_api_key = st.sidebar.text_input('OpenAI API Key')

def count_tokens(string: str) -> int:
	# Load the encoding for gpt-3.5-turbo (which uses cl100k_base)
	encoding_name = "p50k_base"
	encoding = tiktoken.get_encoding(encoding_name) 
	# Encode the input string and count the tokens
	num_tokens = len(encoding.encode(string))
	return num_tokens

def generate_response(input_text):
	try:
		llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
		response = llm(input_text)
		num_tokens = count_tokens(input_text)
		st.info(f"Input contains {num_tokens} tokens.")
		st.info(response)
	except Exception as e:
		st.error(f"An error occurred: {str(e)}")

with st.form('my_form'):
text = st.text_area('Enter text:', 'How to get started with DSA')
submitted = st.form_submit_button('Submit')
if not openai_api_key.startswith('sk-'):
	st.warning('Please enter your OpenAI API key!', icon='')
if submitted and openai_api_key.startswith('sk-'):
	generate_response(text)

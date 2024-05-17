import os
import streamlit as st
import pandas as pd
from pandasai import Agent

# Set your PandasAI API key
os.environ["PANDASAI_API_KEY"] = "$2a$10$jtX1xbqADs4395M3DE9ZcuCx/15PDyWCIe40oLHgyZqEVtvH5lEeG"
# Set a custom cache file path
os.environ["PANDASAI_CACHE"] = r"c:\path\to\new\cache\file.db"

# Create the Streamlit app
st.title("Interactive Sales Data Query")

# File uploader to upload Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Read the uploaded file
    sales_by_country = pd.read_excel(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(sales_by_country.head())

    # Input for user query
    user_query = st.text_input("Enter your query:", "show top 5 Amount by Customer")

    if st.button("Submit"):
        # Create the agent and get the result
        agent = Agent(sales_by_country)
        result = agent.chat(user_query)
        
        # Convert the result to a DataFrame
        result_df = pd.DataFrame(result)

        # Display the result
        st.write("Query Result:")
        st.dataframe(result_df)

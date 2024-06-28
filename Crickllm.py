import streamlit as st
import pandas as pd
import os
from pandasai import SmartDatalake

# Set the PandasAI API key (replace with secure handling in production)
os.environ['PANDASAI_API_KEY'] = "$2a$10$H5wRaaqGlAz9qaXB38HJuusW9W1KzWB71/PJZp/Rs2xJHKO7b4ZoG"

# Function to load data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsb'):
        df = pd.read_excel(uploaded_file, engine='pyxlsb')
    else:
        st.error("Unsupported file format")
        return None
    return df

# Function to generate and display a dynamic plot based on response
def generate_dynamic_plot(response, df):
    # Ensure response is a string
    response_str = str(response)  # Convert response to string
    
    # Example: Check response for keywords and generate corresponding plot
    response_lower = response_str.lower()  # Convert response to lowercase for case-insensitive checks
    
    if 'histogram' in response_lower:
        st.subheader("Histogram")
        selected_column = st.selectbox("Select a column for histogram", df.columns)
        if selected_column:
            st.write("Generating Histogram...")
            st.bar_chart(df[selected_column].value_counts())

    elif 'scatter plot' in response_lower:
        st.subheader("Scatter Plot")
        st.write("Generating Scatter Plot...")
        # Placeholder for scatter plot generation
        # Example: st.scatter_chart(df[selected_column1], df[selected_column2])

    # Add more elif conditions for other types of plots as needed

# Streamlit app
st.title("Data Analysis with PandasAI")

# File upload
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv", "xlsb"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.write("Dataframe Preview:")
        st.dataframe(df.head())

        # Initialize SmartDataframe
        sdf = SmartDatalake(df)

        # Text input for question
        question = st.text_input("Ask a question about your data")
        
        # Button to get answer
        if st.button("Get Answer"):
            if question.strip():  # Check if the question is not empty
                try:
                    response = sdf.chat(question)
                    if response is not None:
                        st.write("Response:")
                        st.write(response)
                        
                        # Generate and display dynamic plot based on response
                        generate_dynamic_plot(response, df)
                        
                    else:
                        st.info("PandasAI did not provide a response.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please enter a question.")
else:
    st.info("Please upload a file to proceed.")

import streamlit as st
import pandas as pd
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv('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')

def generate_analysis(dataframe):
    # Convert dataframe to CSV string
    csv_data = dataframe.to_csv(index=False)

    # Define the prompt for OpenAI GPT
    prompt = f"""
    Analyze the following dataset:
    {csv_data}

    Provide a summary analysis including:
    - Key statistics (mean, median, etc.)
    - Any noticeable trends or patterns
    - Any potential outliers or anomalies
    """

    # Call the OpenAI GPT API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    # Extract the text from the response
    analysis = response.choices[0].text.strip()
    return analysis

# Streamlit UI
st.title("Data Analysis with OpenAI GPT")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Dataset:")
    st.dataframe(df)

    # Generate analysis
    analysis = generate_analysis(df)
    st.write("Analysis:")
    st.text(analysis)
else:
    st.write("Please upload a CSV file to analyze.")

# Streamlit running instructions
st.write("To run this app, use the following command in your terminal:")
st.code("streamlit run app.py")

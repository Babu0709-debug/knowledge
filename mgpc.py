import streamlit as st
from meta_ai_api import MetaAI

# Initialize MetaAI
ai = MetaAI()

# Title of the Streamlit app
st.title("Chatbox with Babu")

# Text input box
user_input = st.text_input("Type your message:")

# Send button
if st.button("click"):
    # Get response from MetaAI
    response = ai.prompt(message=user_input)

    # Print the full response object
    print(response)

    # Check if response is not empty
    if response and response != "":
        # Display the response text
        st.write(response['message'])
    else:
        st.write("No response from Meta AI. Try again!")

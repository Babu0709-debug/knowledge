import streamlit as st
from meta_ai_api import MetaAI


ai = MetaAI()


st.title("Chatbox with Babu")


user_input = st.text_input("Type your message:")

if st.button("click"):
   
    response = ai.prompt(message=user_input)


    print(response)


    if response and response != "":
       
        st.write(response['message'])
    else:
        st.write("No response from Meta AI. Try again!")

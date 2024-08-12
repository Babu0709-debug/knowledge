import streamlit as st
import pyodbc

st.title("ODBC Driver Test for SQL Server (Windows Authentication)")

try:
    # Connection string for Windows Authentication
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=;'
        'DATABASE=;'
        'Trusted_Connection=yes;'
    )
    st.success("Connection Successful!")
except pyodbc.Error as ex:
    st.error(f"Connection failed: {ex}")

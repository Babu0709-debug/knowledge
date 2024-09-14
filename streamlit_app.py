import streamlit as st
import pandas as pd
import pymssql

# Streamlit input widgets for connection details
server_name = st.text_input("Enter SQL Server name", "your_server")
database_name = st.text_input("Enter Database name", "your_database")

# Button to trigger data loading
if st.button('Load Data'):
    try:
        # Establish connection using pymssql with Windows Authentication
        conn = pymssql.connect(
            server=server_name,
            database=database_name
        )

        query = "SELECT * FROM your_table"  # Modify with your table name
        df = pd.read_sql(query, conn)
        conn.close()

        # Display DataFrame in Streamlit
        st.write(df)
    except Exception as e:
        st.error(f"Error: {e}")

import streamlit as st
import pandas as pd
import pyodbc

# Streamlit input widgets for connection details
server_name = st.text_input("Enter SQL Server name", "sestosql05.eu.esab.org")
database_name = st.text_input("Enter Database name", "ODS_live")

# Button to trigger data loading
if st.button('Load Data'):
    try:
        # Establish connection using pyodbc with Windows Authentication
        conn_str = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;'

        # Connect to the database
        conn = pyodbc.connect(conn_str)
        
        # SQL query
        query = "SELECT * FROM your_table"
        
        # Execute query and store result in DataFrame
        df = pd.read_sql(query, conn)
        
        # Close connection
        conn.close()

        # Display DataFrame in Streamlit
        st.write(df)
    
    except Exception as e:
        st.error(f"Error: {e}")

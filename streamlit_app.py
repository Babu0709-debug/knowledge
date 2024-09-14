import streamlit as st
import pyodbc
import pandas as pd

# Define connection parameters
server_name = 'sestosql05.eu.esab.org'
database_name = 'ODS_live'

# Streamlit app layout
st.title('SQL Server Data Viewer')

# Create a text input for SQL query
query = st.text_area('Enter your SQL query', 'SELECT TOP 10 * FROM your_table')

# Button to execute query
if st.button('Execute Query'):
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f'SERVER={server_name};'
        f'DATABASE={database_name};'
        'Trusted_Connection=yes;'
    )
    
    try:
        # Establish connection
        st.info('Connecting to SQL Server...')
        conn = pyodbc.connect(conn_str, timeout=10)  # Set a connection timeout
        # Execute query and load data into DataFrame
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Display the DataFrame
        st.write(df)
    except pyodbc.OperationalError as e:
        st.error(f"Operational Error: {e}")
    except pyodbc.Error as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

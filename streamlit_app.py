import streamlit as st
import pyodbc

# Define connection parameters
server = 'your_server.database.windows.net'
database = 'your_database'
username = 'your_username'
password = 'your_password'
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};"

# Streamlit UI
st.title("SQL Server Connection Test")

try:
    # Establish connection
    conn = pyodbc.connect(conn_str)
    st.write("Connection successful")

    # Test the connection
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 10 * FROM Emos.Sales_invoiced")
    rows = cursor.fetchall()

    # Display the results
    if rows:
        st.write("Query result:")
        for row in rows:
            st.write(row)
    else:
        st.write("No results found.")

except pyodbc.Error as e:
    st.error(f"Failed to connect to SQL Server: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

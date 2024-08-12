import streamlit as st
import pyodbc

# Define connection parameters
server = '10.232.70.46'
database = 'ODS_live'
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"

# Streamlit UI
st.title("SQL Server Connection Test")

try:
    # Establish connection
    conn = pyodbc.connect(conn_str)
    st.write("Connection successful")

    # Test the connection
    cursor = conn.cursor()
    cursor.execute("SELECT top 10 * from Emos.Sales_invoiced")
    rows = cursor.fetchall()

    # Display the results
    if rows:
        st.write("Query result:")
        for row in rows:
            st.write(row)
    else:
        st.write("No results found.")

except Exception as e:
    st.error(f"Failed to connect to SQL Server: {e}")

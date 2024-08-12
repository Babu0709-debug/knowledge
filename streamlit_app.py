import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# Define connection parameters
server = '10.232.70.46'
database = 'ODS_live'
driver = 'ODBC Driver 17 for SQL Server'  # Update this to the correct driver name

# Create SQLAlchemy engine
engine = create_engine(f'mssql+pyodbc://{server}/{database}?driver={driver}')

# Streamlit UI
st.title("SQL Server Connection Test")

try:
    # Establish connection and execute query
    query = "SELECT TOP 10 * FROM Emos.Sales_invoiced"
    df = pd.read_sql(query, engine)
    st.write("Query result:")
    st.write(df)
except Exception as e:
    st.error(f"Failed to connect to SQL Server: {e}")

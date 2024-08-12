import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pyodbc
import warnings

warnings.filterwarnings('ignore')

# Load environment variables if needed
# load_dotenv()

def connect_to_sql_server(server_name, database_name):
    try:
        # Define the connection string for Windows Authentication
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server_name};"
            f"DATABASE={database_name};"
            f"Trusted_Connection=yes;"
        )

        # Create the SQLAlchemy URL
        connection_url = URL.create(
            "mssql+pyodbc",
            query={"odbc_connect": connection_string}
        )

        # Create the SQLAlchemy engine
        engine = create_engine(connection_url)

        # Test the connection
        with engine.connect() as connection:
            df = pd.read_sql_query("SELECT TOP 10 * FROM Information_Schema.Tables", connection)
        return df

    except Exception as e:
        st.error(f"Failed to connect to SQL Server: {e}")
        return None

def main():
    st.set_page_config(page_title="SQL Server Data Fetch", page_icon="📊")
    st.title("SQL Server Data Fetch")

    # Input fields for server and database
    server_name = st.text_input("Server Name", "10.232.70.46")
    database_name = st.text_input("Database Name", "Ods_live")

    # Fetch data from SQL Server
    if server_name and database_name:
        df = connect_to_sql_server(server_name, database_name)
        if df is not None:
            st.success("Successfully fetched data from SQL Server")
            st.dataframe(df)
        else:
            st.warning("No data to display.")
    else:
        st.warning("Please provide both Server Name and Database Name.")

if __name__ == "__main__":
    main()

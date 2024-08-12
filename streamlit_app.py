from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Define the connection string
connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=myserver;DATABASE=test;UID=user;PWD=password"

# Create the SQLAlchemy URL
connection_url = URL.create(
    "mssql+pyodbc", 
    query={"odbc_connect": connection_string}
)

# Create the SQLAlchemy engine
engine = create_engine(connection_url)

# Test the connection
try:
    with engine.connect() as connection:
        result = connection.execute("SELECT 1")
        print("Connection successful, test query result:", result.scalar())
except Exception as e:
    print(f"Failed to connect to SQL Server: {e}")

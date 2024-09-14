import pyodbc

server_name = 'sestosql05.eu.esab.org'
database_name = 'ODS_live'

conn_str = (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    f'SERVER={server_name};'
    f'DATABASE={database_name};'
    'Trusted_Connection=yes;'
)

try:
    conn = pyodbc.connect(conn_str)
    print("Connection successful")
    conn.close()
except pyodbc.Error as e:
    print(f"Error: {e}")

import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

print("Connection parameters:")
print(f"  User: {USER}")
print(f"  Host: {HOST}")
print(f"  Port: {PORT}")
print(f"  Database: {DBNAME}")
print(f"  Password: {'*' * len(PASSWORD) if PASSWORD else 'Not set'}")
print()

# Connect to the database
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("✓ Connection successful!")
    
    # Create a cursor to execute SQL queries
    cursor = connection.cursor()
    
    # Check current time
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print(f"✓ Current Time: {result[0]}")
    
    # Check if tables exist
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('nodes', 'edges')
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    
    if tables:
        print(f"✓ Found tables: {', '.join(t[0] for t in tables)}")
        
        # Count existing records
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table[0]}')
            count = cursor.fetchone()[0]
            print(f"  - {table[0]}: {count} records")
    else:
        print("⚠ No nodes or edges tables found. Make sure tables are created.")
        print("  Check your database schema.")
    
    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("\n✓ Connection closed.")
    
except Exception as e:
    print(f"✗ Failed to connect: {e}")
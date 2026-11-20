import sqlite3
from pathlib import Path

def create_db():
    # create DB folder
    Path("db").mkdir(exist_ok=True)

    # create data base
    conn = sqlite3.connect("db/memory.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    conn.commit()

def main():
    # initialize DB
    create_db()

if __name__ == "__main__":
    main()
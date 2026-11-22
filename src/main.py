import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv(
    "AGENTMEMOS_DB_PATH", str(Path(__file__).resolve().parent.parent / "db" / "memories.db"),
))

def create_messages_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

def create_memory_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            content TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            scope TEXT NOT NULL DEFAULT 'shared',
            embedding TEXT,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)")}

    if "scope" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN scope TEXT NOT NULL DEFAULT 'shared'")

    if "embedding" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN embedding TEXT")

    if "is_deleted" not in columns:
        conn.execute("ALTER TABLE memory ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0")

    # Different agents may store the same key in different namespaces.
    # A key may be reused when an older memory has been soft-deleted.
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_active_memory_key
        ON memory(scope, key) WHERE is_deleted = 0
    """)

def create_memory_history_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            scope TEXT NOT NULL,
            key TEXT NOT NULL,
            content TEXT NOT NULL,
            version INTEGER NOT NULL,
            operation TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

def create_semantic_summary_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            last_message_seq INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

def create_pending_turn_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_memory_turns (
            session_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (session_id, scope)
        )
    """)


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    try:
        create_messages_table(conn)
        create_memory_table(conn)
        create_memory_history_table(conn)
        create_semantic_summary_table(conn)
        create_pending_turn_table(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()

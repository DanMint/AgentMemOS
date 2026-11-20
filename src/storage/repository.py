import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.memory.models import ExtractionContext, Message

class ConversationRepository:
    def __init__(self, db_path="db/memories.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def append_message(self, session_id: str, role: str, content: str) -> int:
        if role not in ("user", "agent"):
            raise ValueError("role must be 'user' or 'agent'")
        
        with self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content)
                   VALUES (?, ?, ?)""",
                (session_id, role, content),
            )
            return cursor.lastrowid

    def get_summary(self, session_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT summary FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            return row["summary"] if row else ""

    def get_recent_messages(self, session_id: str, before_id: int, limit: int = 10) -> list[Message]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ? AND id < ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, before_id, limit),
            ).fetchall()

        return [Message(role=row["role"], content=row["content"])
                for row in reversed(rows)]

    def build_extraction_context(self, session_id: str, user_message_id: int, agent_message_id: int) -> ExtractionContext:
        if user_message_id >= agent_message_id:
            raise ValueError("User message must precede agent message")

        with self.connect() as conn:
            # one consistent snapshot for all context inputs.
            conn.execute("BEGIN")
            pair = conn.execute(
                """SELECT id, role, content FROM messages
                   WHERE session_id = ? AND id IN (?, ?)""",
                (session_id, user_message_id, agent_message_id),
            ).fetchall()

            summary_row = conn.execute(
                "SELECT summary FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            rows = conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ? AND id < ?
                   ORDER BY id DESC LIMIT 10""",
                (session_id, user_message_id),
            ).fetchall()

        by_id = {row["id"]: row for row in pair}

        if user_message_id not in by_id or agent_message_id not in by_id:
            raise ValueError("Message pair not found in session")
        
        user_row, agent_row = by_id[user_message_id], by_id[agent_message_id]

        if user_row["role"] != "user" or agent_row["role"] != "agent":
            raise ValueError("Invalid user-agent message pair")

        return ExtractionContext(
            summary=summary_row["summary"] if summary_row else "",
            recent_messages=[
                Message(role=row["role"], content=row["content"])
                for row in reversed(rows)
            ],
            user_message=Message(role="user", content=user_row["content"]),
            agent_message=Message(role="agent", content=agent_row["content"])
        )
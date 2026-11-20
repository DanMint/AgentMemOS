import sqlite3
from pathlib import Path

from src.memory.models import Message, ExtractionContext

class ConversationRepository:
    def __init__(self, db_path="db/messages.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def append_message(self, session_id, role, content) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages
                (session_id, role, content, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (session_id, role, content)
            )

            return cursor.lastrowid

    def get_summary(self, session_id) -> str:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT summary
                FROM conversation_summaries
                WHERE session_id = ?
                """,
                (session_id,)
            ).fetchone()

            return row["summary"] if row else ""

    def get_recent_messages(self, session_id, before_id, limit=10) -> list[Message]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                  AND id < ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, before_id, limit)
            ).fetchall()

        return [
            Message(role=row["role"], content=row["content"])
            for row in reversed(rows)
        ]

    def build_extraction_context(self, session_id, user_message_id, agent_message_id) -> ExtractionContext:
        with self.connect() as conn:
            pair = conn.execute(
                """
                SELECT id, role, content
                FROM messages
                WHERE session_id = ?
                  AND id IN (?, ?)
                """,
                (
                    session_id,
                    user_message_id,
                    agent_message_id
                )
            ).fetchall()

        by_id = {row["id"]: row for row in pair}

        user_row = by_id[user_message_id]
        agent_row = by_id[agent_message_id]

        if user_row["role"] != "user" or agent_row["role"] != "agent":
            raise ValueError("Invalid user-agent message pair")

        return ExtractionContext(
            summary=self.get_summary(session_id),

            recent_messages=self.get_recent_messages(session_id=session_id, before_id=user_message_id, limit=10),

            user_message=Message(role="user", content=user_row["content"]),

            agent_message=Message(role="agent", content=agent_row["content"])
        )
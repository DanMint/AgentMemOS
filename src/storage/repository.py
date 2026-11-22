import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.memory.models import ExtractionContext, Message

RECENT_MESSAGE_LIMIT = 10


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

    def get_pending_memory_turn(self, session_id: str, scope: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM pending_memory_turns WHERE session_id=? AND scope=?",
                (session_id, scope),
            ).fetchone()
        return row["payload"] if row else None

    def save_pending_memory_turn(self, session_id: str, scope: str, payload: str | None) -> None:
        with self.connect() as conn:
            if payload is None:
                conn.execute(
                    "DELETE FROM pending_memory_turns WHERE session_id=? AND scope=?",
                    (session_id, scope),
                )
            else:
                conn.execute(
                    """INSERT INTO pending_memory_turns (session_id, scope, payload)
                       VALUES (?, ?, ?) ON CONFLICT(session_id, scope) DO UPDATE
                       SET payload=excluded.payload""",
                    (session_id, scope, payload),
                )

    def get_summary(self, session_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT summary FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            return row["summary"] if row else ""

    def get_summary_batch(
        self, session_id: str, through_id: int, batch_size: int,
    ) -> tuple[str, int, int, list[Message]] | None:
        with self.connect() as conn:
            conn.execute("BEGIN")
            state = conn.execute(
                """SELECT summary, last_message_seq FROM conversation_summaries
                   WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
            checkpoint = state["last_message_seq"] if state else 0
            required = batch_size + RECENT_MESSAGE_LIMIT
            rows = conn.execute(
                """SELECT id, role, content FROM messages
                   WHERE session_id = ? AND id > ? AND id <= ?
                   ORDER BY id LIMIT ?""",
                (session_id, checkpoint, through_id, required),
            ).fetchall()

        if len(rows) < required:
            return None
        batch = rows[:batch_size]
        messages = [Message(role=row["role"], content=row["content"]) for row in batch]
        return state["summary"] if state else "", checkpoint, batch[-1]["id"], messages

    def save_summary(self, session_id: str, summary: str,
                     checkpoint: int, through_id: int) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO conversation_summaries (session_id, summary, last_message_seq)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       summary = excluded.summary,
                       last_message_seq = excluded.last_message_seq,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE conversation_summaries.last_message_seq = ?""",
                (session_id, summary, through_id, checkpoint),
            )
            return cursor.rowcount == 1

    def get_recent_messages(self, session_id: str, before_id: int, limit: int = RECENT_MESSAGE_LIMIT) -> list[Message]:
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
                """SELECT summary, last_message_seq FROM conversation_summaries
                   WHERE session_id = ? AND last_message_seq < ?""",
                (session_id, user_message_id),
            ).fetchone()

            rows = conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ? AND id < ? AND id > ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, user_message_id,
                 summary_row["last_message_seq"] if summary_row else 0,
                 RECENT_MESSAGE_LIMIT),
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
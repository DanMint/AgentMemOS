import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.memory.identity import same_fact
from src.memory.models import MemoryCandidate, MemoryRecord, PreparedMemoryChange, UpdateResult


class MemoryConflictError(Exception):
    """An optimistic write failed or a key already exists."""


class MemoryRepository:
    def __init__(self, db_path="db/memories.db"):
        self.db_path = Path(db_path)

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

    @staticmethod
    def _record(row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], key=row["key"], content=row["content"],
            version=row["version"], scope=row["scope"],
        )

    def list_active(self, scope: str) -> list[tuple[MemoryRecord, list[float] | None]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM memory
                   WHERE scope = ? AND is_deleted = 0 ORDER BY id""",
                (scope,),
            ).fetchall()
        return [
            (self._record(row), json.loads(row["embedding"]) if row["embedding"] else None)
            for row in rows
        ]

    def add_memory(self, key: str, content: str, scope: str,
                   embedding: list[float]) -> MemoryRecord:
        try:
            with self.connect() as conn:
                cursor = conn.execute(
                    """INSERT INTO memory (key, content, scope, embedding)
                       VALUES (?, ?, ?, ?)""",
                    (key, content, scope, json.dumps(embedding)),
                )
                row = conn.execute(
                    "SELECT * FROM memory WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                conn.execute(
                    """INSERT INTO memory_history
                       (memory_id, scope, key, content, version, operation)
                       VALUES (?, ?, ?, ?, 1, 'ADD')""",
                    (row["id"], scope, key, content),
                )
                return self._record(row)
        except sqlite3.IntegrityError as exc:
            raise MemoryConflictError("Active memory key already exists") from exc

    def update_memory(self, memory_id: int, expected_version: int,
                      scope: str, content: str,
                      embedding: list[float]) -> MemoryRecord:
        with self.connect() as conn:
            cursor = conn.execute(
                """UPDATE memory
                   SET content = ?, embedding = ?, version = version + 1,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND scope = ? AND version = ? AND is_deleted = 0""",
                (content, json.dumps(embedding), memory_id, scope, expected_version),
            )
            if cursor.rowcount != 1:
                raise MemoryConflictError("Memory changed or was deleted")
            row = conn.execute(
                "SELECT * FROM memory WHERE id = ?", (memory_id,)
            ).fetchone()
            conn.execute(
                """INSERT INTO memory_history
                   (memory_id, scope, key, content, version, operation)
                   VALUES (?, ?, ?, ?, ?, 'UPDATE')""",
                (memory_id, scope, row["key"], row["content"], row["version"]),
            )
            return self._record(row)

    def delete_memory(self, memory_id: int, expected_version: int,
                      scope: str) -> MemoryRecord:
        with self.connect() as conn:
            cursor = conn.execute(
                """UPDATE memory
                   SET is_deleted = 1, version = version + 1,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND scope = ? AND version = ? AND is_deleted = 0""",
                (memory_id, scope, expected_version),
            )
            if cursor.rowcount != 1:
                raise MemoryConflictError("Memory changed or was deleted")
            row = conn.execute(
                "SELECT * FROM memory WHERE id = ?", (memory_id,)
            ).fetchone()
            conn.execute(
                """INSERT INTO memory_history
                   (memory_id, scope, key, content, version, operation)
                   VALUES (?, ?, ?, ?, ?, 'DELETE')""",
                (memory_id, scope, row["key"], row["content"], row["version"]),
            )
            return self._record(row)


    def apply_batch(self, changes: list[PreparedMemoryChange],
                    scope: str) -> list[UpdateResult]:
        """Atomically commit all operations from one user interaction.

        BEGIN IMMEDIATE serializes competing SQLite writers; each UPDATE/DELETE
        still checks its expected version to detect stale LLM decisions.
        """
        results: list[UpdateResult] = []
        try:
            with self.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for change in changes:
                    if change.action == "ADD":
                        if change.content is None or change.embedding is None:
                            raise ValueError("ADD requires content and embedding")
                        if change.key.startswith(("user.likes.", "user.dislikes.")):
                            candidate = MemoryCandidate(key=change.key, content=change.content)
                            rows = conn.execute(
                                """SELECT id, key, content, version, scope FROM memory
                                   WHERE scope=? AND is_deleted=0
                                     AND (key LIKE 'user.likes.%' OR key LIKE 'user.dislikes.%')""",
                                (scope,),
                            )
                            if any(same_fact(candidate, self._record(row)) for row in rows):
                                raise MemoryConflictError("An active preference for this subject already exists")
                        cursor = conn.execute(
                            """INSERT INTO memory (key, content, scope, embedding)
                               VALUES (?, ?, ?, ?)""",
                            (change.key, change.content, scope,
                             json.dumps(change.embedding)),
                        )
                        memory_id = cursor.lastrowid
                        version = 1
                        content = change.content
                    elif change.action in ("UPDATE", "DELETE"):
                        if change.target_id is None or change.expected_version is None:
                            raise ValueError("UPDATE/DELETE require ID and version")
                        if change.action == "UPDATE":
                            if change.content is None or change.embedding is None:
                                raise ValueError("UPDATE requires content and embedding")
                            cursor = conn.execute(
                                """UPDATE memory
                                   SET key=?, content=?, embedding=?, version=version+1,
                                       updated_at=CURRENT_TIMESTAMP
                                   WHERE id=? AND scope=? AND version=? AND is_deleted=0""",
                                (change.key, change.content, json.dumps(change.embedding),
                                 change.target_id, scope, change.expected_version),
                            )
                        else:
                            cursor = conn.execute(
                                """UPDATE memory
                                   SET is_deleted=1, version=version+1,
                                       updated_at=CURRENT_TIMESTAMP
                                   WHERE id=? AND scope=? AND version=? AND is_deleted=0""",
                                (change.target_id, scope, change.expected_version),
                            )
                        if cursor.rowcount != 1:
                            raise MemoryConflictError("Memory changed or was deleted")
                        row = conn.execute(
                            "SELECT id, key, content, version FROM memory WHERE id=?",
                            (change.target_id,),
                        ).fetchone()
                        memory_id = row["id"]
                        version = row["version"]
                        content = row["content"]
                    else:
                        raise ValueError(f"Unsupported operation {change.action}")

                    row = conn.execute("SELECT key FROM memory WHERE id=?",
                                       (memory_id,)).fetchone()
                    key = row["key"]
                    conn.execute(
                        """INSERT INTO memory_history
                           (memory_id, scope, key, content, version, operation)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (memory_id, scope, key, content, version, change.action),
                    )
                    results.append(UpdateResult(
                        action=change.action, key=key,
                        memory_id=memory_id, version=version,
                    ))
        except sqlite3.IntegrityError as exc:
            raise MemoryConflictError("Active memory key already exists") from exc
        return results

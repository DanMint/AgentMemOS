import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.main import create_memory_history_table, create_memory_table
from src.memory.consistency import MemoryConsistency
from src.memory.manager import MemoryUpdateManager
from src.memory.models import ExtractedMemories, MemoryCandidate
from src.storage.memory_repository import MemoryRepository


class StubRetriever:
    async def find_similar(self, **kwargs):
        return []

    async def embed_document(self, content):
        return [1.0]


class InvestmentMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "memory.db"
        with sqlite3.connect(path) as connection:
            create_memory_table(connection)
            create_memory_history_table(connection)
        self.repository = MemoryRepository(path)
        self.manager = MemoryUpdateManager(
            StubRetriever(), None, MemoryConsistency(self.repository),
        )

    async def test_assistant_goal_accepts_want_and_preserves_investment_tracking(self):
        extracted = ExtractedMemories(memories=[MemoryCandidate(
            key="user.financial_assistant", content="The user wants a personal financial assistant.",
        )])
        added, = await self.manager.process(
            extracted,
            source_message="I want you to be my personal financial assistant who keeps track of my investments",
        )
        record, _ = self.repository.list_active("shared")[0]
        self.assertEqual((added.action, added.memory_id), ("ADD", record.id))
        self.assertIn("keeps track of their investments", record.content)

    async def test_deposit_advances_from_in_progress_to_complete_once(self):
        extracted = ExtractedMemories(memories=[MemoryCandidate(
            key="user.deposit", content="The user deposited $50,000.",
        )])
        pending_message = "Im depositing 50000$ to start investing"
        pending, = await self.manager.process(extracted, source_message=pending_message)
        record, _ = self.repository.list_active("shared")[0]
        self.assertEqual(pending.action, "ADD")
        self.assertEqual(record.content, "The user is depositing $50000 for investments.")

        completed, = await self.manager.process(
            extracted, source_message="I have deposited $50,000 for investments.",
        )
        self.assertEqual((completed.action, completed.memory_id, completed.version),
                         ("UPDATE", pending.memory_id, 2))
        repeated, = await self.manager.process(extracted, source_message=pending_message)
        self.assertEqual((repeated.action, repeated.version), ("NOOP", 2))
        record, _ = self.repository.list_active("shared")[0]
        self.assertEqual(record.content, "The user deposited $50000 for investments.")

    async def test_stock_purchase_ignores_old_facts_and_invented_balances(self):
        sale = self.repository.add_memory(
            "user.vehicle.history.sale", "The user sold their car for 40000.", "shared", [1.0],
        )
        extracted = ExtractedMemories(memories=[
            MemoryCandidate(key="user.vehicle", content="The user's vehicle is a Corolla."),
            MemoryCandidate(key="user.vehicle.sale_proceeds", content="The user received $40,000."),
            MemoryCandidate(key="user.investment.remaining", content="The user has $80,000 remaining."),
            MemoryCandidate(key="user.investment.stocks.value", content="The user's 4 stocks were worth $10,000."),
        ])
        source = "i bought 4 stocks which were worth 10000$"
        added, = await self.manager.process(extracted, source_message=source)
        self.assertEqual(added.action, "ADD")
        active = {record.key: record for record, _ in self.repository.list_active("shared")}
        self.assertEqual(set(active), {sale.key, added.key})
        self.assertEqual(active[added.key].content, "The user bought 4 stocks worth $10000.")
        repeated, = await self.manager.process(extracted, source_message=source)
        self.assertEqual((repeated.action, repeated.memory_id), ("NOOP", added.memory_id))

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.main import create_memory_history_table, create_memory_table
from src.memory.consistency import MemoryConsistency
from src.memory.manager import MemoryUpdateManager
from src.memory.models import ExtractedMemories, MemoryCandidate, VehiclePurchase
from src.storage.memory_repository import MemoryRepository


class StubRetriever:
    async def find_similar(self, **kwargs):
        return []

    async def embed_document(self, content):
        return [1.0]


class VehiclePurchaseTests(unittest.IsolatedAsyncioTestCase):
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
        self.original = self.repository.add_memory(
            "user.vehicle", "The user's vehicle is a 2012 Lexus.",
            "shared", [1.0],
        )

    async def purchase(self, source, *, memories=None, **purchase):
        extracted = ExtractedMemories(
            vehicle_purchase=VehiclePurchase(**purchase), memories=memories or [],
        )
        return await self.manager.process(extracted, source_message=source)

    def active(self):
        return {record.key: record for record, _ in self.repository.list_active("shared")}

    async def test_adds_another_car_with_different_wording_and_replays_cleanly(self):
        source = "I just purchased a second vehicle: a green 2020 Camry SE for $20,000."
        fields = dict(model="Camry SE", year=2020, color="green", price="$20,000")
        added, = await self.purchase(source, **fields, memories=[
            MemoryCandidate(key="user.vehicle", content="Camry SE"),
            MemoryCandidate(key="user.vehicle.color", content="green"),
        ])
        self.assertEqual((added.action, added.key), ("ADD", "user.vehicle.additional.2020.camry.se"))
        self.assertEqual(self.active()["user.vehicle"], self.original)
        self.assertIn("$20000", self.active()[added.key].content)
        repeated, = await self.purchase(source, **fields)
        self.assertEqual((repeated.action, repeated.memory_id), ("NOOP", added.memory_id))

    async def test_updates_current_vehicle_without_delete(self):
        updated, = await self.purchase(
            "I purchased a BMW 3 Series from 2019 for 30,000.",
            model="BMW 3 Series", year=2019, price="30000",
        )
        self.assertEqual((updated.action, updated.memory_id), ("UPDATE", self.original.id))
        self.assertIn("2019 BMW 3 Series", self.active()["user.vehicle"].content)

    async def test_ignores_plans_and_unsupported_fields(self):
        for source, fields in [
            ("I plan to buy a 2020 Camry.", dict(model="Camry", year=2020)),
            ("I just purchased a 2020 Camry.", dict(model="Toyota Camry", year=2020)),
            ("I bought a 2020 Camry for $20,000.", dict(model="Camry", year=2020, price="$2000")),
        ]:
            with self.subTest(source=source, fields=fields):
                self.assertEqual(await self.purchase(source, **fields), [])
        self.assertEqual(self.active(), {"user.vehicle": self.original})

    async def test_supports_reported_wording_when_model_does_not_emit_event(self):
        extracted = ExtractedMemories(memories=[
            MemoryCandidate(key="user.vehicle", content="The user's new car is a green 2020 Camry SE."),
        ])
        added, = await self.manager.process(
            extracted,
            source_message="I have spent 20000$ on another car. The car its a green 2020 camry se",
        )
        self.assertEqual((added.action, added.key), ("ADD", "user.vehicle.additional.2020.camry.se"))
        self.assertEqual(self.active()["user.vehicle"], self.original)

    async def test_just_sold_car_retires_details_and_saves_proceeds(self):
        color = self.repository.add_memory(
            "user.vehicle.color", "The user's vehicle is black.", "shared", [1.0],
        )
        deposit = self.repository.add_memory(
            "user.deposit", "The user deposited $50,000.", "shared", [1.0],
        )
        extracted = ExtractedMemories(memories=[MemoryCandidate(
            key="user.vehicle_sale_proceeds",
            content="The user sold their car for $40,000.",
        )])
        source = "I just sold my car and got 40000 for it"
        changes = await self.manager.process(extracted, source_message=source)
        self.assertEqual(
            {change.memory_id for change in changes if change.action == "DELETE"},
            {self.original.id, color.id},
        )
        active = self.active()
        self.assertEqual(active["user.vehicle.history.sale"].content, "The user sold their car for 40000.")
        self.assertEqual(active[deposit.key], deposit)
        self.assertEqual(await self.manager.process(extracted, source_message=source), [])

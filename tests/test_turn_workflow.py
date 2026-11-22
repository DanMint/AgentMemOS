import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.agents.turn_workflow import build_turn_workflow
from src.main import create_memory_history_table, create_memory_table, create_pending_turn_table
from src.memory.event_store import EventStore, PendingTurn
from src.memory.events import TurnInterpretation
from src.storage.memory_repository import MemoryRepository
from src.storage.repository import ConversationRepository


class Interpreter:
    async def interpret(self, message, recent, pending_vehicle, feedback=None):
        if feedback and message in self.repairs:
            return self.repairs[message]
        return self.turns[message]


class Retriever:
    async def embed_document(self, content):
        return [1.0]


class Agent:
    async def communicate(self, message, recent, summary, memories):
        return "\n".join(memories) or "Hello."


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "memory.db"
        with sqlite3.connect(path) as db:
            create_memory_table(db)
            create_memory_history_table(db)
            create_pending_turn_table(db)
        self.repository = MemoryRepository(path)
        self.conversation = ConversationRepository(path)
        self.interpreter = Interpreter()
        self.interpreter.turns = {}
        self.interpreter.repairs = {}
        self.graph = build_turn_workflow(
            self.interpreter, EventStore(self.repository, Retriever(), "shared"), Agent(),
        )
        self.turn_id = 0

    async def turn(self, text, *events, pending=None, balance=False):
        self.turn_id += 1
        self.interpreter.turns[text] = TurnInterpretation.model_validate({
            "events": events, "asks_for_balance": balance,
        })
        return await self.graph.ainvoke({
            "message": text, "turn_id": str(self.turn_id), "recent": [],
            "summary": "", "pending": pending,
        })

    def active(self):
        return {record.key: record.content for record, _ in self.repository.list_active("shared")}

    async def test_paraphrased_sale_with_other_fact_commits_as_one_turn(self):
        purchase = "I picked up a white 2012 Lexus with a V6 for 20 grand."
        result = await self.turn(purchase, dict(
            kind="vehicle_purchased", evidence=purchase,
            description="a white 2012 Lexus with a V6", amount="20 grand",
        ))
        self.assertEqual([op.action for op in result["operations"]], ["ADD"])
        self.repository.add_memory("user.vehicle.color", "The user's vehicle is white.", "shared", [1.])
        sale = "The Lexus went for 14 grand"
        down_payment = "I put a down payment on a new house"
        result = await self.turn(f"{sale} and {down_payment}.",
            dict(kind="vehicle_sold", evidence=sale, vehicle="Lexus", amount="14 grand"),
            dict(kind="fact", evidence=down_payment, key="user.house.down_payment",
                 content="The user put a down payment on a new house."),
        )
        self.assertEqual([op.action for op in result["operations"]], ["DELETE", "DELETE", "ADD", "ADD"])
        self.assertNotIn("user.vehicle", self.active())
        self.assertIn("14 grand", result["response"])
        self.assertIn("down payment", result["response"])

    async def test_ambiguous_sale_waits_and_named_followup_resolves_one_car(self):
        self.repository.add_memory("user.vehicle", "The user's vehicle is a 2019 BMW.", "shared", [1.])
        purchase = "I spent 20000$ on another car, a green 2020 Camry SE."
        added = await self.turn(purchase, dict(kind="vehicle_purchased", evidence=purchase,
            description="a green 2020 Camry SE", amount="20000$"))
        self.assertEqual(added["operations"][0].action, "ADD")
        sale = "I just sold my car and got 40000 for it"
        unclear = await self.turn(sale, dict(kind="vehicle_sold", evidence=sale, amount="40000"))
        self.assertIn("Which vehicle", unclear["response"])
        self.assertEqual(len(self.active()), 2)
        pending = unclear["pending"]
        self.conversation.save_pending_memory_turn("session", "shared", pending.model_dump_json())
        persisted = self.conversation.get_pending_memory_turn("session", "shared")
        resolved = await self.turn("The Camry", dict(kind="vehicle_selection", evidence="The Camry",
            vehicle="Camry"), pending=PendingTurn.model_validate_json(persisted))
        self.assertIsNone(resolved["pending"])
        self.assertIn("2020 Camry", resolved["response"])
        self.assertTrue(any("2019 BMW" in value for value in self.active().values()))
        deleted = [op.key for op in resolved["operations"] if op.action == "DELETE"]
        self.assertEqual(deleted, [added["operations"][0].key])

    async def test_deposit_stock_purchase_and_read_only_balance(self):
        pending = "I'm putting 50,000$ into investments"
        first = await self.turn(pending, dict(kind="deposit", evidence=pending,
            amount="50,000$", destination="investments", status="pending"))
        cleared = "The $50,000 investment deposit cleared"
        second = await self.turn(cleared, dict(kind="deposit", evidence=cleared,
            amount="$50,000", status="completed"))
        self.assertEqual(first["operations"][0].memory_id, second["operations"][0].memory_id)
        self.assertEqual(second["operations"][0].action, "UPDATE")
        stocks = "I bought four stocks which were worth 10000$"
        bought = await self.turn(stocks, dict(kind="stock_purchase", evidence=stocks,
            quantity="four stocks", amount="10000$", amount_kind="value"))
        self.assertIn("four stocks worth $10000", bought["response"])
        before = self.active()
        query = await self.turn("How much do I have left to invest?", balance=True)
        self.assertEqual(before, self.active())
        self.assertIn("don't have a confirmed current balance", query["response"])

    async def test_unverified_detail_rolls_back_the_entire_turn(self):
        self.repository.add_memory("user.vehicle", "The user's vehicle is a Lexus.", "shared", [1.])
        sale = "I sold the Lexus for $14,000"
        deposit = "I deposited $500"
        result = await self.turn(f"{sale} and {deposit}.",
            dict(kind="vehicle_sold", evidence=sale, vehicle="Lexus", amount="$14,000"),
            dict(kind="deposit", evidence=deposit, amount="$50", status="completed"),
        )
        self.assertIn("No memory changes were saved", result["response"])
        self.assertEqual(set(self.active()), {"user.vehicle"})

    async def test_repairs_ungrounded_extraction_before_commit(self):
        source = "I bought 4 stocks for $10,000."
        self.interpreter.repairs[source] = TurnInterpretation.model_validate({
            "events": [{"kind": "stock_purchase", "evidence": source,
                        "quantity": "4 stocks", "amount": "$10,000"}],
        })
        result = await self.turn(source, dict(
            kind="stock_purchase", evidence=source, quantity="Not stated",
        ))
        self.assertEqual(result["attempts"], 2)
        self.assertEqual([op.action for op in result["operations"]], ["ADD"])
        self.assertIn("costing $10000", result["response"])

    async def test_confirmation_can_resolve_a_single_pending_deposit_without_restating_amount(self):
        pending = "I'm putting $500 into investments."
        started = await self.turn(pending, dict(kind="deposit", evidence=pending,
            amount="$500", destination="investments", status="pending"))
        cleared = "It cleared."
        confirmed = await self.turn(cleared, dict(kind="deposit", evidence=cleared, status="completed"))
        self.assertEqual(
            (confirmed["operations"][0].action, confirmed["operations"][0].memory_id),
            ("UPDATE", started["operations"][0].memory_id),
        )
        self.assertIn("deposited $500 for investments", confirmed["response"])

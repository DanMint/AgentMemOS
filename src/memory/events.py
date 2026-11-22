"""Interpret the user's turn as facts and events, before generating a reply."""

from typing import Annotated, Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.memory.models import Message


class UserEvent(BaseModel):
    evidence: str = Field(min_length=1, description="Exact excerpt from the latest user message")


class VehiclePurchased(UserEvent):
    kind: Literal["vehicle_purchased"]
    description: str = Field(min_length=1, description="Vehicle identity, quoted from the message")
    amount: str | None = Field(default=None, description="Price as spoken, including any currency")
    replaces: str | None = Field(default=None, description="Explicitly replaced vehicle reference")


class VehicleRemoved(UserEvent):
    kind: Literal["vehicle_sold", "vehicle_lost"]
    vehicle: str | None = Field(default=None, description="Named vehicle, if stated in this turn")
    amount: str | None = None


class VehicleDetail(UserEvent):
    kind: Literal["vehicle_detail"]
    vehicle: str | None = None
    attribute: Literal["color", "year", "horsepower", "transmission", "engine", "trim", "make", "model"]
    value: str = Field(min_length=1, description="New value quoted from the latest message")


class VehicleSelection(UserEvent):
    kind: Literal["vehicle_selection"]
    vehicle: str = Field(min_length=1, description="Vehicle named in this clarification")


class Deposit(UserEvent):
    kind: Literal["deposit"]
    amount: str | None = Field(default=None, min_length=1)
    destination: str | None = None
    status: Literal["pending", "completed"]


class StockPurchase(UserEvent):
    kind: Literal["stock_purchase"]
    quantity: str = Field(min_length=1)
    amount: str | None = None
    security: str | None = None
    amount_kind: Literal["cost", "value"] = "cost"


class Fact(UserEvent):
    kind: Literal["fact"]
    key: str = Field(description="Stable descriptive key for a new fact or correction")
    content: str


Event = Annotated[
    VehiclePurchased | VehicleRemoved | VehicleDetail | VehicleSelection | Deposit | StockPurchase | Fact,
    Field(discriminator="kind"),
]


class TurnInterpretation(BaseModel):
    events: list[Event] = Field(default_factory=list)
    asks_for_balance: bool = False


INTERPRETATION_PROMPT = """
Interpret the latest user message as structured events. Treat earlier user messages
only as reference context. Every event requires an exact excerpt from the latest
message as evidence. Copy descriptions, names, values, quantities, and amounts
from that excerpt, including the original currency and units. Leave unknown fields null.
Extract all independently asserted events in the order they happened.

Questions, recaps, hypothetical statements, greetings, and plans do not establish
completed events. A question alone gives an empty events list. Set asks_for_balance
for questions about available funds, cash, or a remaining investment balance.
Do not infer balances or account links from transactions.

Use these event meanings:
- vehicle_purchased: the user actually acquired a vehicle. Include the name/model,
  and any color or year stated as part of its description. Set replaces only when
  the user explicitly says the new vehicle replaces an existing one. Buying
  another vehicle does not establish that an earlier vehicle was disposed of.
- vehicle_sold or vehicle_lost: the user explicitly disposed of or no longer
  possesses a vehicle. A crash alone does not establish loss. Leave vehicle null
  for references like "my car" and "it". Leave price null when it was not stated.
- vehicle_selection: the user is answering a pending question about which vehicle
  a prior event targets. Quote their selection in vehicle. Do not re-extract the prior event.
- vehicle_detail: the user states or corrects a vehicle detail. Use one event per
  attribute. Supply a vehicle name only if stated in this turn.
- deposit: record a pending or completed deposit as stated, preserving the amount.
  A deposit being made is pending. Confirmation that it cleared is completed.
  For confirmation of an earlier deposit without an amount, leave amount null.
- stock_purchase: the user bought stocks or shares. Keep stated cost distinct from
  stated value. Do not infer tickers or amounts.
- fact: other durable user facts or preferences, goals, and corrections. Do not use
  fact for vehicle transactions/attributes, financial transactions, or balances
  inferred from arithmetic. A directly stated account balance may be a fact.
  Use user.vehicle for an explicit current vehicle identity without a purchase.
  Preserve independent facts and use stable keys.

For fact keys use user.* or project.*. Write the fact as a complete declarative
sentence, preserving its action and outcome.

Examples of complete interpretations:
"The Lexus went for 14 grand and I put a down payment on a new house"
{"events":[{"kind":"vehicle_sold","evidence":"The Lexus went for 14 grand","vehicle":"Lexus","amount":"14 grand"},{"kind":"fact","evidence":"I put a down payment on a new house","key":"user.house.down_payment","content":"The user put a down payment on a new house."}],"asks_for_balance":false}

"I bought four stocks worth 10000$"
{"events":[{"kind":"stock_purchase","evidence":"I bought four stocks worth 10000$","quantity":"four stocks","amount":"10000$","amount_kind":"value"}],"asks_for_balance":false}

"I'm depositing 50000$ to start investing"
{"events":[{"kind":"deposit","evidence":"I'm depositing 50000$ to start investing","amount":"50000$","destination":"investing","status":"pending"}],"asks_for_balance":false}

"How much do I have left to invest?"
{"events":[],"asks_for_balance":true}

"""


class TurnInterpreter:
    def __init__(self, model: str):
        self.model = ChatOllama(model=model, temperature=0, reasoning=False).with_structured_output(
            TurnInterpretation, method="json_schema",
        )

    async def interpret(self, message: str, recent: list[Message],
                        pending_vehicle: bool = False,
                        feedback: str | None = None) -> TurnInterpretation:
        history = "\n".join(item.content for item in recent if item.role == "user")
        context = (f"Pending vehicle clarification: {pending_vehicle}\n"
                   f"Earlier user messages:\n{history}\nLatest user message:\n{message}")
        if feedback:
            context += f"\nValidation rejected your last interpretation:\n{feedback}"
        result = await self.model.ainvoke([
            ("system", INTERPRETATION_PROMPT),
            ("human", context),
        ])
        if result is None:
            raise ValueError("The interpreter returned no structured result")
        return result

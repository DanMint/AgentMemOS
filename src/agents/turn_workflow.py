"""Explicit interpretation, persistence, and response stages for the CLI."""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.memory.consistency import MemoryValidationError
from src.memory.event_store import EventStore, InvalidEvidence, PendingTurn, validate_vehicle_selection
from src.memory.events import TurnInterpretation, VehicleSelection
from src.memory.models import Message, UpdateResult
from src.storage.memory_repository import MemoryConflictError


class TurnState(TypedDict):
    message: str
    turn_id: str
    recent: list[Message]
    summary: str
    pending: PendingTurn | None
    interpretation: NotRequired[TurnInterpretation]
    operations: NotRequired[list[UpdateResult]]
    response: NotRequired[str | None]
    error: NotRequired[str | None]
    feedback: NotRequired[str | None]
    attempts: NotRequired[int]


def build_turn_workflow(interpreter, store: EventStore, agent):
    async def interpret(state: TurnState) -> dict:
        try:
            result = await interpreter.interpret(
                state["message"], state["recent"], state["pending"] is not None,
                feedback=state.get("feedback"),
            )
            return {
                "interpretation": result, "error": None, "response": None,
                "feedback": None, "attempts": state.get("attempts", 0) + 1,
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "response": "I couldn't interpret this turn for memory. No memory changes were saved.",
            }

    async def save(state: TurnState) -> dict:
        current = state["interpretation"]
        pending = state["pending"]
        selection = next((event for event in current.events if isinstance(event, VehicleSelection)), None)
        interpretation, source, turn_id, selections = current, state["message"], state["turn_id"], {}
        try:
            if selection:
                if not pending:
                    raise MemoryValidationError("There is no vehicle update waiting for clarification.")
                validate_vehicle_selection(selection, source)
                selected = store.select_vehicle(pending, selection)
                selections = {**pending.selections, pending.event_index: selected}
                interpretation = pending.interpretation.model_copy(update={
                    "events": pending.interpretation.events + [
                        event for event in current.events if not isinstance(event, VehicleSelection)
                    ],
                })
                source = pending.source + "\n" + source
                turn_id = pending.turn_id
            if not interpretation.events:
                return {}
            operations, details, clarification = await store.apply(
                interpretation, source, turn_id, selections,
            )
        except InvalidEvidence as exc:
            if state["attempts"] == 1 and not selection:
                return {"feedback": f"{exc} Re-examine this output: {current.model_dump_json()}"}
            return {"error": str(exc), "response": f"{exc} No memory changes were saved."}
        except (MemoryValidationError, MemoryConflictError, ValueError) as exc:
            return {"error": str(exc), "response": f"{exc} No memory changes were saved."}
        except Exception as exc:
            return {
                "error": str(exc),
                "response": "I couldn't save this turn. No memory changes were committed.",
            }

        if clarification:
            options = "\n".join(f"- {detail}" for detail in details)
            return {"pending": clarification, "response": f"Which vehicle did you mean?\n{options}"}

        result = {"operations": operations, "pending": None if selection else pending}
        if operations:
            saved = "\n".join(f"- {detail}" for detail in details)
            result["response"] = f"Saved:\n{saved}" if saved else "Updated your saved information."
            if current.asks_for_balance:
                result["response"] += "\nYour current balance is unknown."
        elif interpretation.events:
            result["response"] = "That information is already recorded."
            if current.asks_for_balance:
                result["response"] += " Your current balance is unknown."
        return result

    async def reply(state: TurnState) -> dict:
        if state.get("response"):
            return {}
        if state["interpretation"].asks_for_balance:
            return {"response": "I don't have a confirmed current balance. Individual deposits, sales, and purchases do not establish it."}
        memories = [record.content for record, _ in store.repository.list_active(store.scope)]
        try:
            response = await agent.communicate(state["message"], state["recent"], state["summary"], memories)
        except Exception as exc:
            return {"response": "I couldn't generate a reply. No memory changes were made.", "error": str(exc)}
        return {"response": response}

    async def after_interpretation(state: TurnState) -> str:
        return "reply" if state.get("error") else "save"

    async def after_save(state: TurnState) -> str:
        return "interpret" if state.get("feedback") else "reply"

    graph = StateGraph(TurnState)
    graph.add_node("interpret", interpret)
    graph.add_node("save", save)
    graph.add_node("reply", reply)
    graph.add_edge(START, "interpret")
    graph.add_conditional_edges("interpret", after_interpretation,
                                {"reply": "reply", "save": "save"})
    graph.add_conditional_edges("save", after_save,
                                {"interpret": "interpret", "reply": "reply"})
    graph.add_edge("reply", END)
    return graph.compile()

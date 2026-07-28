"""Deterministic execution for unambiguous, state-changing workplace requests.

LLMs remain responsible for conversation and ambiguous requests. This guardrail
ensures that clear operational commands always call an approved domain tool.
"""

import re

from langchain_core.messages import AIMessage

from src.tools.finance_tools import check_reimbursement_status
from src.tools.hr_tools import check_leave_balance, holiday_calendar
from src.tools.travel_tools import estimate_budget
from src.utils.logging_config import record_event


def _employee_id(text: str) -> str | None:
    match = re.search(r"\bEMP\d+\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def _expense_id(text: str) -> str | None:
    match = re.search(r"\bEXP\d+\b", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def dispatch_known_request(route: str, text: str) -> AIMessage | None:
    """Return a trusted tool result for a request whose required inputs are clear."""
    lowered = text.lower()
    employee_id = _employee_id(text)

    # All IT state-changing actions (password reset, unlock, tickets) go through
    # the human-in-loop state machine in app.py. Nothing dispatched here for IT.

    if route == "hr_agent" and employee_id and "leave" in lowered and any(
        word in lowered for word in ("balance", "remaining", "how many")
    ):
        record_event("tool_requested", agent=route, tool="check_leave_balance")
        return AIMessage(content=check_leave_balance.invoke({"employee_id": employee_id}))
    if route == "hr_agent" and "holiday" in lowered:
        record_event("tool_requested", agent=route, tool="holiday_calendar")
        return AIMessage(content=holiday_calendar.invoke({}))

    expense_id = _expense_id(text)
    if route == "finance_agent" and employee_id and expense_id and any(
        word in lowered for word in ("status", "reimbursement", "claim")
    ):
        record_event("tool_requested", agent=route, tool="check_reimbursement_status")
        result = check_reimbursement_status(expense_id, employee_id)
        if not result["success"]:
            return AIMessage(content=result["message"])
        return AIMessage(
            content=(
                f"Expense {result['expense_id']} is {result['status']}.\n"
                f"Amount: {result['currency']} {result['amount']:,.2f}\n"
                f"Remarks: {result['remarks']}"
            )
        )

    if route == "travel_agent" and "budget" in lowered:
        match = re.search(r"\b(\d+)\s*days?\b", lowered)
        if match:
            from src.utils.paths import TRAVEL_RATES_FILE
            from src.utils.file_handler import load_json

            destinations = load_json(TRAVEL_RATES_FILE)
            destination = next(
                (city for city in destinations if city.lower() in lowered), None
            )
            if destination:
                record_event("tool_requested", agent=route, tool="estimate_budget")
                return AIMessage(
                    content=estimate_budget.invoke(
                        {"destination": destination, "days": int(match.group(1))}
                    )
                )
    return None

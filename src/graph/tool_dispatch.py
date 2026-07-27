"""Deterministic execution for unambiguous, state-changing workplace requests.

LLMs remain responsible for conversation and ambiguous requests. This guardrail
ensures that clear operational commands always call an approved domain tool.
"""

import re

from langchain_core.messages import AIMessage

from src.tools.finance_tools import check_reimbursement_status
from src.tools.hr_tools import check_leave_balance, holiday_calendar
from src.tools.it_tools import reset_password, unlock_account
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

    if route == "it_agent" and "reset" in lowered and "password" in lowered and not employee_id:
        record_event("tool_requested", agent=route, tool="reset_password", status="missing_employee_id")
        return AIMessage(content="Please provide your employee ID (for example, EMP001) so I can submit the password reset request.")
    if route == "it_agent" and employee_id and "reset" in lowered and "password" in lowered:
        record_event("tool_requested", agent=route, tool="reset_password")
        return AIMessage(content=reset_password.invoke({"employee_id": employee_id}))
    if route == "it_agent" and employee_id and "unlock" in lowered and "account" in lowered:
        record_event("tool_requested", agent=route, tool="unlock_account")
        return AIMessage(content=unlock_account.invoke({"employee_id": employee_id}))

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

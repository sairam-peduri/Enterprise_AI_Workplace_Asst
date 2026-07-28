"""Context manager that gathers employee data for personalized recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.paths import EMPLOYEE_FILE, TICKET_FILE, FINANCE_DATA_DIR, HR_DATA_DIR, TRAVEL_FILE
from src.utils.file_handler import load_json


class ContextManager:
    """Gathers and aggregates employee context from existing enterprise data."""

    def get_employee_context(self, employee_id: str) -> dict[str, Any]:
        """Build a complete context dictionary for an employee."""
        employee = self._find_employee(employee_id)
        if not employee:
            return {"employee_id": employee_id, "found": False}
        return {
            "employee_id": employee_id,
            "found": True,
            "name": employee.get("name", ""),
            "department": employee.get("department", ""),
            "email": employee.get("email", ""),
            "role": self._infer_role(employee),
            "open_tickets": self._get_open_tickets(employee_id),
            "leave_balance": self._get_leave_balance(employee_id),
            "pending_expenses": self._get_pending_expenses(employee_id),
            "travel_requests": self._get_travel_requests(employee_id),
            "recent_activity": [],
        }

    def _find_employee(self, employee_id: str) -> dict | None:
        try:
            employees = load_json(EMPLOYEE_FILE)
            return next((e for e in employees if e.get("employee_id") == employee_id), None)
        except Exception:
            return None

    def _infer_role(self, employee: dict) -> str:
        dept = employee.get("department", "").lower()
        if dept == "it":
            return "Technical"
        if dept == "finance":
            return "Finance"
        if dept == "hr":
            return "HR"
        if dept == "sales":
            return "Sales"
        return "General"

    def _get_open_tickets(self, employee_id: str) -> list[dict]:
        try:
            tickets = load_json(TICKET_FILE)
            return [t for t in tickets if t.get("employee_id") == employee_id and t.get("status") != "Resolved"]
        except Exception:
            return []

    def _get_leave_balance(self, employee_id: str) -> dict[str, Any]:
        try:
            leaves = load_json(HR_DATA_DIR / "leave.json")
            emp_leaves = [l for l in leaves if l.get("employee_id") == employee_id and l.get("status") == "Approved"]
            total_taken = sum(l.get("days", 0) for l in emp_leaves)
            return {"total_entitled": 20, "taken": total_taken, "remaining": 20 - total_taken}
        except Exception:
            return {"total_entitled": 20, "taken": 0, "remaining": 20}

    def _get_pending_expenses(self, employee_id: str) -> list[dict]:
        try:
            expenses = load_json(FINANCE_DATA_DIR / "expenses.json")
            return [e for e in expenses if e.get("employee_id") == employee_id and e.get("status") not in ("Reimbursed", "Rejected")]
        except Exception:
            return []

    def _get_travel_requests(self, employee_id: str) -> list[dict]:
        try:
            requests_data = load_json(TRAVEL_FILE)
            return [r for r in requests_data if r.get("employee_id") == employee_id]
        except Exception:
            return []

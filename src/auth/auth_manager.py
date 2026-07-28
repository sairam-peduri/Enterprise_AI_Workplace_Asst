"""Authentication and role-based access control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.paths import EMPLOYEE_FILE, HR_DATA_DIR

EMPLOYEE_DB = Path(EMPLOYEE_FILE)
LEAVE_DB = HR_DATA_DIR / "leave.json"
EXPENSE_DB = Path(__file__).resolve().parents[2] / "data" / "finance" / "expenses.json"
TICKET_DB = Path(__file__).resolve().parents[2] / "data" / "it" / "tickets.json"
TRAVEL_DB = Path(__file__).resolve().parents[2] / "data" / "travel" / "travel_requests.json"


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []


def _save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class AuthManager:
    """Handles login, roles, and data mutations for requests."""

    def __init__(self):
        self.employees = _load_json(EMPLOYEE_DB)

    def get_employee(self, employee_id: str) -> dict[str, Any] | None:
        return next((e for e in self.employees if e.get("employee_id") == employee_id), None)

    def validate_password(self, employee_id: str, password: str) -> bool:
        """Validate employee password."""
        emp = self.get_employee(employee_id)
        if not emp:
            return False
        if emp.get("account_locked", False):
            return False
        return emp.get("password", "") == password

    def is_account_locked(self, employee_id: str) -> bool:
        """Check if employee account is locked."""
        emp = self.get_employee(employee_id)
        return emp.get("account_locked", False) if emp else True

    def get_employee_name(self, employee_id: str) -> str:
        emp = self.get_employee(employee_id)
        return emp["name"] if emp else employee_id

    def get_employee_department(self, employee_id: str) -> str:
        emp = self.get_employee(employee_id)
        return emp.get("department", "") if emp else ""

    def is_hr(self, employee_id: str) -> bool:
        return self.get_employee_department(employee_id).upper() == "HR"

    def is_manager(self, employee_id: str) -> bool:
        return self.is_hr(employee_id)

    def get_all_employees(self) -> list[dict]:
        return self.employees

    def get_all_employee_ids(self) -> list[str]:
        return [e.get("employee_id", "") for e in self.employees]

    def get_pending_leave_requests(self) -> list[dict]:
        leaves = _load_json(LEAVE_DB)
        return [l for l in leaves if l.get("status") == "Pending"]

    def get_pending_leave_for_employee(self, employee_id: str) -> list[dict]:
        leaves = _load_json(LEAVE_DB)
        return [l for l in leaves if l.get("employee_id") == employee_id and l.get("status") == "Pending"]

    def get_approved_leave_for_employee(self, employee_id: str) -> list[dict]:
        leaves = _load_json(LEAVE_DB)
        return [l for l in leaves if l.get("employee_id") == employee_id and l.get("status") == "Approved"]

    def submit_leave_request(self, employee_id: str, days: int, leave_type: str, reason: str) -> dict:
        leaves = _load_json(LEAVE_DB)
        new_request = {
            "employee_id": employee_id,
            "employee_name": self.get_employee_name(employee_id),
            "days": days,
            "leave_type": leave_type,
            "reason": reason,
            "status": "Pending",
        }
        leaves.append(new_request)
        _save_json(LEAVE_DB, leaves)
        return new_request

    def approve_leave(self, employee_id: str, days: int, leave_type: str, reason: str) -> bool:
        leaves = _load_json(LEAVE_DB)
        for leave in leaves:
            if (leave.get("employee_id") == employee_id
                    and leave.get("days") == days
                    and leave.get("leave_type") == leave_type
                    and leave.get("reason") == reason
                    and leave.get("status") == "Pending"):
                leave["status"] = "Approved"
                _save_json(LEAVE_DB, leaves)
                return True
        return False

    def decline_leave(self, employee_id: str, days: int, leave_type: str, reason: str) -> bool:
        leaves = _load_json(LEAVE_DB)
        for leave in leaves:
            if (leave.get("employee_id") == employee_id
                    and leave.get("days") == days
                    and leave.get("leave_type") == leave_type
                    and leave.get("reason") == reason
                    and leave.get("status") == "Pending"):
                leave["status"] = "Rejected"
                _save_json(LEAVE_DB, leaves)
                return True
        return False

    def get_pending_expenses(self) -> list[dict]:
        expenses = _load_json(EXPENSE_DB)
        return [e for e in expenses if e.get("status") == "Pending"]

    def get_pending_expenses_for_employee(self, employee_id: str) -> list[dict]:
        expenses = _load_json(EXPENSE_DB)
        return [e for e in expenses if e.get("employee_id") == employee_id and e.get("status") == "Pending"]

    def approve_expense(self, expense_id: str) -> bool:
        expenses = _load_json(EXPENSE_DB)
        for exp in expenses:
            if exp.get("expense_id") == expense_id and exp.get("status") == "Pending":
                exp["status"] = "Approved"
                _save_json(EXPENSE_DB, expenses)
                return True
        return False

    def decline_expense(self, expense_id: str) -> bool:
        expenses = _load_json(EXPENSE_DB)
        for exp in expenses:
            if exp.get("expense_id") == expense_id and exp.get("status") == "Pending":
                exp["status"] = "Rejected"
                _save_json(EXPENSE_DB, expenses)
                return True
        return False

    def get_pending_tickets(self) -> list[dict]:
        tickets = _load_json(TICKET_DB)
        return [t for t in tickets if t.get("status") not in ("Resolved", "Closed")]

    def get_pending_tickets_for_employee(self, employee_id: str) -> list[dict]:
        tickets = _load_json(TICKET_DB)
        return [t for t in tickets if t.get("employee_id") == employee_id and t.get("status") not in ("Resolved", "Closed")]

    def get_pending_travel(self) -> list[dict]:
        travels = _load_json(TRAVEL_DB)
        return [t for t in travels if t.get("status") == "Pending"]

    def get_pending_travel_for_employee(self, employee_id: str) -> list[dict]:
        travels = _load_json(TRAVEL_DB)
        return [t for t in travels if t.get("employee_id") == employee_id and t.get("status") == "Pending"]

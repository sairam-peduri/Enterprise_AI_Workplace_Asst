import re
from difflib import SequenceMatcher

from langchain_core.tools import tool

from src.utils.file_handler import load_json, save_json
from src.utils.paths import HR_DATA_DIR

HR_EMPLOYEE_FILE = HR_DATA_DIR / "employees.json"
HR_LEAVE_FILE = HR_DATA_DIR / "leave.json"
HR_HOLIDAY_FILE = HR_DATA_DIR / "holidays.json"


def _load_employees():
    return load_json(HR_EMPLOYEE_FILE)


def _load_leaves():
    return load_json(HR_LEAVE_FILE)


def _load_holidays():
    return load_json(HR_HOLIDAY_FILE)


def _save_leaves(leaves):
    save_json(HR_LEAVE_FILE, leaves)


def _find_employee(employee_id, employees):
    employee_id = employee_id.strip().upper()
    return next(
        (emp for emp in employees if str(emp.get("employee_id", "")).upper() == employee_id),
        None,
    )


def find_employee_by_name(employee_name: str) -> dict | None:
    """Find one employee by full name, ignoring case and extra spaces."""
    normalized_name = " ".join(employee_name.split()).casefold()
    if not normalized_name:
        return None
    return next(
        (
            employee
            for employee in _load_employees()
            if " ".join(str(employee.get("name", "")).split()).casefold() == normalized_name
        ),
        None,
    )


def find_employee_in_text(text: str) -> dict | None:
    """Resolve a known full employee name mentioned in a chat message."""
    normalized_text = " ".join(text.split()).casefold()
    return next(
        (
            employee
            for employee in _load_employees()
            if employee.get("name")
            and " ".join(str(employee["name"]).split()).casefold() in normalized_text
        ),
        None,
    )


def suggest_employee_in_text(text: str) -> dict | None:
    """Suggest a close full-name match without treating it as an exact identity."""
    words = re.findall(r"[a-z]+", text.casefold())
    best_match: dict | None = None
    best_score = 0.0
    for employee in _load_employees():
        name_words = re.findall(r"[a-z]+", str(employee.get("name", "")).casefold())
        if not name_words:
            continue
        window_size = len(name_words)
        for index in range(len(words) - window_size + 1):
            candidate = " ".join(words[index : index + window_size])
            score = SequenceMatcher(None, candidate, " ".join(name_words)).ratio()
            if score > best_score:
                best_match = employee
                best_score = score
    return best_match if best_score >= 0.84 else None


def prepare_leave_application(
    employee_name: str, days: int, leave_type: str, reason: str
) -> dict:
    """Validate a leave request without changing the mock HR data."""
    employee = find_employee_by_name(employee_name)
    if employee is None:
        return {"success": False, "message": "Employee not found."}
    if days <= 0:
        return {"success": False, "message": "Leave duration must be at least one day."}
    if not reason.strip():
        return {"success": False, "message": "A reason for leave is required."}
    balance = int(employee.get("leave_balance", 0))
    if balance < days:
        return {
            "success": False,
            "message": f"Leave request cannot be submitted. {employee['name']} has only {balance} day(s) available.",
        }
    return {
        "success": True,
        "employee": employee,
        "message": "Leave request is ready for confirmation.",
    }


@tool
def apply_leave(
    employee_id: str,
    days: int,
    leave_type: str,
    reason: str,
    confirmed: bool = False,
) -> str:
    """
    Apply for leave and reduce the employee's available balance.

    This tool changes mock HR data only when confirmed is True.
    """
    if days <= 0:
        return "Leave duration must be at least one day."
    if not leave_type.strip() or not reason.strip():
        return "Leave type and reason are required."
    if not confirmed:
        return "Confirmation required. Show the leave summary and ask the employee to reply yes before submitting."
    employee_id = employee_id.strip().upper()
    employees = _load_employees()
    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    balance = int(employee.get("leave_balance", 0))
    if balance < days:
        return f"Leave request denied. Only {balance} days are available."

    employee["leave_balance"] = balance - days
    leaves = _load_leaves()
    leave_request = {
        "employee_id": employee_id,
        "employee_name": employee.get("name", employee_id),
        "days": days,
        "leave_type": leave_type,
        "reason": reason,
        "status": "Pending",
    }
    leaves.append(leave_request)
    _save_leaves(leaves)
    save_json(HR_EMPLOYEE_FILE, employees)

    return (
        f"Leave request submitted successfully for {employee.get('name', employee_id)}. "
        f"{days} day(s) requested. Remaining balance: {employee['leave_balance']} days."
    )


@tool
def check_leave_balance(employee_id: str) -> str:
    """
    Check the current leave balance for an employee.
    """
    employee_id = employee_id.strip().upper()
    employees = _load_employees()
    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    balance = employee.get("leave_balance", 0)
    return f"{employee.get('name', employee_id)} has {balance} days of leave remaining."


@tool
def holiday_calendar() -> str:
    """
    List upcoming public holidays from the HR holiday calendar.
    """
    holidays = _load_holidays()
    if not holidays:
        return "No holidays found in the calendar."

    formatted = [f"- {holiday.get('date', 'N/A')}: {holiday.get('name', 'Holiday')}" for holiday in holidays]
    return "Holiday calendar:\n" + "\n".join(formatted)


HR_TOOLS = [apply_leave, check_leave_balance, holiday_calendar]

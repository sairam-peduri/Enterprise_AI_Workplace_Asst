from pathlib import Path

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
    return next(
        (emp for emp in employees if emp.get("employee_id") == employee_id),
        None,
    )


@tool
def apply_leave(employee_id: str, days: int, leave_type: str, reason: str) -> str:
    """
    Apply for leave and reduce the employee's available balance.
    """
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

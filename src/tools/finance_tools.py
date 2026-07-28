import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from langchain_core.tools import StructuredTool

from src.utils.file_handler import load_json, save_json
from src.utils.paths import FINANCE_DATA_DIR, IT_DATA_DIR

DATA_FILE: Path = FINANCE_DATA_DIR / "expenses.json"
FINANCE_EMPLOYEE_FILE: Path = IT_DATA_DIR / "employees.json"


def _load_employees():
    return load_json(FINANCE_EMPLOYEE_FILE)


def load_expenses():
    return load_json(DATA_FILE)


def save_expenses(expenses):
    save_json(DATA_FILE, expenses)


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
    """Resolve a known full employee name or employee ID mentioned in a chat message."""
    normalized_text = " ".join(text.split()).casefold()
    id_match = re.search(r"\b(EMP\d+)\b", text, flags=re.IGNORECASE)
    if id_match:
        emp_id = id_match.group(1).upper()
        return next(
            (emp for emp in _load_employees() if emp.get("employee_id", "").upper() == emp_id),
            None,
        )
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


def check_reimbursement_status(expense_id: str, employee_id: str):
    """
    Check reimbursement status.

    employee_id is required so one employee cannot access
    another employee's expense.
    """

    expenses = load_expenses()

    for expense in expenses:
        if (
            expense["expense_id"].upper() == expense_id.strip().upper()
            and expense["employee_id"].upper() == employee_id.strip().upper()
        ):
            return {
                "success": True,
                "expense_id": expense["expense_id"],
                "amount": expense["amount"],
                "currency": expense["currency"],
                "category": expense["category"],
                "description": expense["description"],
                "status": expense["status"],
                "approved_amount": expense["approved_amount"],
                "remarks": expense["remarks"],
            }

    return {
        "success": False,
        "message": "Expense claim not found or you do not have access to it.",
    }


def generate_expense_id(expenses):
    """Generate the next expense ID."""

    numbers = []

    for expense in expenses:
        expense_id = expense.get("expense_id", "")

        if expense_id.startswith("EXP"):
            try:
                numbers.append(int(expense_id[3:]))
            except ValueError:
                continue

    return f"EXP{max(numbers, default=1000) + 1}"


def submit_expense(
    employee_id: str,
    amount: float,
    category: str,
    description: str,
    expense_date: str,
    receipt_available: bool,
    confirmed: bool = False,
):
    """Submit a new expense claim. Only submits when confirmed is True."""

    if not confirmed:
        return {
            "success": False,
            "message": "Confirmation required. Show the expense summary and ask the employee to reply yes before submitting.",
        }

    if not employee_id.strip():
        return {"success": False, "message": "Employee ID is required."}

    if amount <= 0:
        return {
            "success": False,
            "message": "Expense amount must be greater than zero.",
        }

    if not category.strip():
        return {"success": False, "message": "Expense category is required."}

    if not description.strip():
        return {"success": False, "message": "Description is required."}

    expenses = load_expenses()

    expense_id = generate_expense_id(expenses)

    if receipt_available:
        status = "Pending Manager Approval"
        remarks = "Submitted and awaiting manager approval"
    else:
        status = "Action Required"
        remarks = "Supporting receipt or invoice is required"

    new_expense = {
        "expense_id": expense_id,
        "employee_id": employee_id.strip().upper(),
        "amount": float(amount),
        "currency": "INR",
        "category": category.strip(),
        "description": description.strip(),
        "expense_date": expense_date,
        "submitted_date": date.today().isoformat(),
        "receipt_available": receipt_available,
        "status": status,
        "approved_amount": None,
        "remarks": remarks,
    }

    expenses.append(new_expense)
    save_expenses(expenses)

    return {
        "success": True,
        "message": "Expense claim submitted successfully.",
        "expense": new_expense,
    }


check_reimbursement_tool = StructuredTool.from_function(
    func=check_reimbursement_status,
    name="check_reimbursement_status",
    description=(
        "Check the reimbursement status of an employee's expense claim. "
        "Requires the expense ID and the authenticated employee ID."
    ),
)


FINANCE_TOOLS = [
    check_reimbursement_tool,
]

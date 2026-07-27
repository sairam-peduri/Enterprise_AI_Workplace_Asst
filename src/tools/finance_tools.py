import json
from datetime import date
from pathlib import Path


DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "finance"
    / "expenses.json"
)


def load_expenses():
    """Load all expense records."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_expenses(expenses):
    """Save expense records."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(expenses, file, indent=2)


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
):
    """Submit a new expense claim."""

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

from langchain_core.tools import StructuredTool


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
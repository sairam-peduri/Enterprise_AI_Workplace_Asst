from src.tools.finance_tools import submit_expense


def confirm_expense_submission(
    confirmed: bool,
    employee_id: str,
    amount: float,
    category: str,
    description: str,
    expense_date: str,
    receipt_available: bool,
):
    """
    Submit an expense only after explicit user confirmation.
    """

    if not confirmed:
        return {
            "success": False,
            "submitted": False,
            "message": "Expense submission cancelled.",
        }

    result = submit_expense(
        employee_id=employee_id,
        amount=amount,
        category=category,
        description=description,
        expense_date=expense_date,
        receipt_available=receipt_available,
    )

    return {
        **result,
        "submitted": result.get("success", False),
    }
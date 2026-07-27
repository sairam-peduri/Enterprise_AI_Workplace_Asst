from src.tools.finance_tools import (
    check_reimbursement_status,
    submit_expense,
)


print("\n--- VALID STATUS CHECK ---")

print(
    check_reimbursement_status(
        expense_id="EXP1002",
        employee_id="EMP001",
    )
)


print("\n--- UNAUTHORIZED STATUS CHECK ---")

print(
    check_reimbursement_status(
        expense_id="EXP1003",
        employee_id="EMP001",
    )
)


print("\n--- EXPENSE SUBMISSION ---")

print(
    submit_expense(
        employee_id="EMP001",
        amount=850,
        category="Local Travel",
        description="Cab from office to client meeting",
        expense_date="2026-07-27",
        receipt_available=True,
    )
)
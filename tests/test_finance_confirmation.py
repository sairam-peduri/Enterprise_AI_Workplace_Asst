from src.utils.finance_confirmation import confirm_expense_submission


def test_cancel_submission():
    result = confirm_expense_submission(
        confirmed=False,
        employee_id="EMP001",
        amount=1200,
        category="Local Travel",
        description="Cab to client office",
        expense_date="2026-07-26",
        receipt_available=True,
    )

    print("\n--- CANCEL TEST ---")
    print(result)


def test_confirm_submission():
    result = confirm_expense_submission(
        confirmed=True,
        employee_id="EMP001",
        amount=1500,
        category="Meals",
        description="Lunch during client meeting",
        expense_date="2026-07-26",
        receipt_available=True,
    )

    print("\n--- CONFIRM TEST ---")
    print(result)


if __name__ == "__main__":
    test_cancel_submission()
    test_confirm_submission()
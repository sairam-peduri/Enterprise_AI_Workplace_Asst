import streamlit as st

from src.utils.finance_confirmation import confirm_expense_submission


def initialize_finance_state():
    """
    Initialize Finance-related Streamlit session state.
    """

    if "pending_expense" not in st.session_state:
        st.session_state.pending_expense = None


def set_pending_expense(
    employee_id: str,
    amount: float,
    category: str,
    description: str,
    expense_date: str,
    receipt_available: bool,
):
    """
    Store an expense temporarily while waiting
    for the employee to confirm or cancel.
    """

    st.session_state.pending_expense = {
        "employee_id": employee_id,
        "amount": amount,
        "category": category,
        "description": description,
        "expense_date": expense_date,
        "receipt_available": receipt_available,
    }


def render_expense_confirmation():
    """
    Display Confirm / Cancel controls for a pending expense.
    """

    expense = st.session_state.get("pending_expense")

    if not expense:
        return

    st.markdown("### Expense Summary")

    st.write(f"**Employee ID:** {expense['employee_id']}")
    st.write(f"**Amount:** ₹{expense['amount']:,.2f}")
    st.write(f"**Category:** {expense['category']}")
    st.write(f"**Description:** {expense['description']}")
    st.write(f"**Expense Date:** {expense['expense_date']}")

    receipt_text = (
        "Yes"
        if expense["receipt_available"]
        else "No"
    )

    st.write(f"**Receipt Available:** {receipt_text}")

    confirm_col, cancel_col = st.columns(2)

    with confirm_col:

        if st.button(
            "Confirm Expense",
            type="primary",
            use_container_width=True,
        ):

            result = confirm_expense_submission(
                confirmed=True,
                **expense,
            )

            if result["success"]:

                expense_record = result["expense"]

                st.success(
                    f"Expense {expense_record['expense_id']} "
                    f"submitted successfully."
                )

                st.session_state.pending_expense = None

            else:

                st.error(result["message"])

    with cancel_col:

        if st.button(
            "Cancel",
            use_container_width=True,
        ):

            result = confirm_expense_submission(
                confirmed=False,
                **expense,
            )

            st.info(result["message"])

            st.session_state.pending_expense = None
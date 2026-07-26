from typing import Literal

from state.state import EnterpriseState


def supervisor_node(
    state: EnterpriseState,
) -> Literal[
    "it_agent",
    "hr_agent",
    "finance_agent",
    "travel_agent",
    "knowledge_agent",
]:
    """
    Determines which department should handle the user's request.
    """

    user_message = state["messages"][-1].content.lower()

    # IT
    if any(
        word in user_message
        for word in [
            "password",
            "account",
            "vpn",
            "software",
            "ticket",
            "laptop",
            "system",
        ]
    ):
        return "it_agent"

    # HR
    if any(
        word in user_message
        for word in [
            "leave",
            "holiday",
            "employee",
            "attendance",
        ]
    ):
        return "hr_agent"

    # Finance
    if any(
        word in user_message
        for word in [
            "expense",
            "salary",
            "reimbursement",
            "invoice",
        ]
    ):
        return "finance_agent"

    # Travel
    if any(
        word in user_message
        for word in [
            "travel",
            "hotel",
            "flight",
            "booking",
        ]
    ):
        return "travel_agent"

    return "knowledge_agent"
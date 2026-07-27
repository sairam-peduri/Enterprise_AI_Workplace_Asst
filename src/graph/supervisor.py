"""Deterministic routing for the enterprise specialist agents."""

from typing import Literal

from src.state.state import EnterpriseState
from src.utils.logging_config import record_event

Route = Literal["general_agent", "it_agent", "hr_agent", "finance_agent", "travel_agent", "knowledge_agent"]
ROUTES: dict[str, tuple[str, ...]] = {
    "it_agent": ("password", "account", "vpn", "software", "ticket", "laptop", "system", "hardware", "computer", "unlock"),
    "finance_agent": ("expense", "salary", "reimbursement", "invoice", "claim"),
    "travel_agent": ("travel", "hotel", "flight", "booking", "trip", "itinerary"),
    "hr_agent": ("leave", "holiday", "attendance", "employee", "pto"),
}
GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}


def supervisor_node(state: EnterpriseState) -> Route:
    """Route the latest request; the UI never selects an agent."""
    messages = state.get("messages", [])
    if not messages:
        record_event("route_selected", route="general_agent", reason="empty_conversation")
        return "general_agent"
    user_message = str(getattr(messages[-1], "content", "")).strip().lower()
    if user_message in GREETING_WORDS or user_message in {"help", "what can you do", "who are you"}:
        record_event("route_selected", route="general_agent", reason="greeting_or_help")
        return "general_agent"
    for route, keywords in ROUTES.items():
        if any(keyword in user_message for keyword in keywords):
            record_event("route_selected", route=route, reason="keyword_match")
            return route  # type: ignore[return-value]
    record_event("route_selected", route="knowledge_agent", reason="knowledge_fallback")
    return "knowledge_agent"

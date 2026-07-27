"""Single-file Streamlit chat interface for Enterprise AI."""

from __future__ import annotations

from datetime import datetime
import re
from uuid import uuid4

import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.supervisor import supervisor_node
from src.graph.workflow import workflow
from src.tools.hr_tools import (
    apply_leave,
    find_employee_in_text,
    prepare_leave_application,
    suggest_employee_in_text,
)
from src.utils.logging_config import LOG_FILE, capture_activity, record_event


st.set_page_config(
    page_title="Enterprise AI Workplace Assistant",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

AGENT_LABELS = {
    "general_agent": "Enterprise AI",
    "it_agent": "IT Support",
    "hr_agent": "HR",
    "finance_agent": "Finance",
    "travel_agent": "Travel",
    "knowledge_agent": "Knowledge",
}


def _new_session() -> dict:
    """Create the in-browser record for one independent conversation."""
    return {
        "id": str(uuid4()),
        "title": "New conversation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
        "activity": [],
        "pending_leave": None,
    }


def initialize_sessions() -> None:
    """Initialize session-local conversations and select the first one."""
    if "chat_sessions" not in st.session_state:
        first_session = _new_session()
        st.session_state.chat_sessions = {first_session["id"]: first_session}
        st.session_state.active_session_id = first_session["id"]
    else:
        for session in st.session_state.chat_sessions.values():
            session.setdefault("activity", [])
            session.setdefault("pending_leave", None)


def active_session() -> dict:
    return st.session_state.chat_sessions[st.session_state.active_session_id]


def start_new_session() -> None:
    session = _new_session()
    st.session_state.chat_sessions[session["id"]] = session
    st.session_state.active_session_id = session["id"]


def _session_title(prompt: str) -> str:
    compact_prompt = " ".join(prompt.split())
    return f"{compact_prompt[:42]}{'...' if len(compact_prompt) > 42 else ''}" or "New conversation"


@st.cache_data(ttl=10, show_spinner=False)
def ollama_is_available() -> bool:
    try:
        return requests.get("http://127.0.0.1:11434/api/tags", timeout=1).ok
    except requests.RequestException:
        return False


def render_sidebar() -> None:
    """Render session navigation; no agent selection is exposed to the user."""
    with st.sidebar:
        st.title("Enterprise AI")
        st.caption("Workplace Assistant")

        if st.button("+ New chat", use_container_width=True, type="primary"):
            start_new_session()
            st.rerun()

        st.divider()
        st.caption("YOUR SESSIONS")
        sessions = list(st.session_state.chat_sessions.values())
        sessions.sort(key=lambda item: item["created_at"], reverse=True)
        for session in sessions:
            is_active = session["id"] == st.session_state.active_session_id
            if st.button(
                session["title"],
                key=f"session-{session['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_session_id = session["id"]
                st.rerun()

        st.divider()
        if len(sessions) > 1 and st.button("Delete current chat", use_container_width=True):
            deleted_id = st.session_state.active_session_id
            del st.session_state.chat_sessions[deleted_id]
            st.session_state.active_session_id = next(iter(st.session_state.chat_sessions))
            st.rerun()

        st.caption("Messages are retained separately for each open session.")
        with st.expander("Activity log", expanded=False):
            activity = active_session()["activity"]
            if activity:
                for item in activity:
                    details = ", ".join(
                        f"{key}={value}" for key, value in item.items() if key not in {"timestamp", "event"}
                    )
                    st.caption(f"{item['timestamp']} | {item['event']} | {details}")
            else:
                st.caption("No activity recorded for this session yet.")
            st.caption(f"Persistent log file: {LOG_FILE}")
        if ollama_is_available():
            st.success("Ollama connected")
        else:
            st.warning("Ollama unavailable")


def render_messages(messages: list) -> None:
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(str(message.content))
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                agent = message.additional_kwargs.get("agent", "Enterprise AI")
                st.caption(f"Routed to {agent}")
                st.markdown(str(message.content))


def _leave_details(prompt: str) -> tuple[int | None, str | None, str]:
    """Extract leave details from a natural-language follow-up."""
    days_match = re.search(r"\b(\d+)\s*days?\b", prompt, flags=re.IGNORECASE)
    reason_match = re.search(r"\breason\s*(?:is|:|=)?\s*(.+)", prompt, flags=re.IGNORECASE)
    if not reason_match:
        reason_match = re.search(r"\b(?:because|due to)\s+(.+)", prompt, flags=re.IGNORECASE)
    leave_type = "Sick Leave" if re.search(r"\b(sick|fever|medical)\b", prompt, flags=re.IGNORECASE) else "Annual Leave"
    return (int(days_match.group(1)) if days_match else None, reason_match.group(1).strip() if reason_match else None, leave_type)


def _leave_confirmation_message(leave: dict) -> str:
    return (
        "Please confirm this leave request:\n\n"
        f"- Employee: {leave['employee_name']} ({leave['employee_id']})\n"
        f"- Duration: {leave['days']} day(s)\n"
        f"- Type: {leave['leave_type']}\n"
        f"- Reason: {leave['reason']}\n\n"
        "Reply **yes** to submit it, or **no** to cancel."
    )


def handle_hr_leave_request(session: dict, prompt: str) -> str | None:
    """Run the name-based, explicit-confirmation HR leave flow for one session."""
    pending = session.get("pending_leave")
    normalized_prompt = prompt.strip().casefold()

    if pending and pending["stage"] == "awaiting_employee_confirmation":
        suggested_employee = pending["employee"]
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            session["pending_leave"] = {"stage": "collecting_details", "employee": suggested_employee}
            record_event("hr_employee_suggestion_accepted", employee_id=suggested_employee["employee_id"])
            return (
                f"Employee confirmed: {suggested_employee['name']} ({suggested_employee['employee_id']}). "
                "Please provide the number of days and a reason."
            )
        if normalized_prompt in {"no", "n", "cancel"}:
            session["pending_leave"] = None
            record_event("hr_employee_suggestion_rejected")
            return "Employee not found. Please provide the employee's full name exactly as recorded in HR."
        return "Please reply **yes** to use the suggested employee or **no** to cancel."

    if pending and pending["stage"] == "awaiting_confirmation":
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            result = apply_leave.invoke(
                {
                    "employee_id": pending["employee_id"],
                    "days": pending["days"],
                    "leave_type": pending["leave_type"],
                    "reason": pending["reason"],
                    "confirmed": True,
                }
            )
            session["pending_leave"] = None
            result_text = str(result)
            record_event(
                "hr_leave_submitted" if result_text.startswith("Leave request submitted successfully") else "hr_leave_submission_failed",
                employee_id=pending["employee_id"],
                days=pending["days"],
            )
            return result_text
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            session["pending_leave"] = None
            record_event("hr_leave_cancelled", employee_id=pending["employee_id"])
            return "Leave request cancelled. No HR data was changed."
        return "Please reply **yes** to submit the leave request or **no** to cancel it."

    employee = pending.get("employee") if pending else find_employee_in_text(prompt)
    is_leave_request = pending is not None or "leave" in normalized_prompt or "time off" in normalized_prompt
    if not is_leave_request:
        return None
    if employee is None:
        suggested_employee = suggest_employee_in_text(prompt)
        if suggested_employee is not None:
            session["pending_leave"] = {"stage": "awaiting_employee_confirmation", "employee": suggested_employee}
            record_event("hr_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
            return (
                f"Employee not found. Did you mean **{suggested_employee['name']}** "
                f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
            )
        record_event("hr_employee_not_found")
        return "Employee not found. Please provide the employee's full name exactly as recorded in HR."

    days, reason, leave_type = _leave_details(prompt)
    if days is None or reason is None:
        session["pending_leave"] = {"stage": "collecting_details", "employee": employee}
        record_event("hr_leave_details_requested", employee_id=employee["employee_id"])
        return (
            f"Employee found: {employee['name']} ({employee['employee_id']}). "
            "Please provide the number of days and a reason, for example: `5 days, reason: fever`."
        )

    prepared = prepare_leave_application(employee["name"], days, leave_type, reason)
    if not prepared["success"]:
        session["pending_leave"] = None
        record_event("hr_leave_validation_failed", employee_id=employee["employee_id"])
        return prepared["message"]

    session["pending_leave"] = {
        "stage": "awaiting_confirmation",
        "employee_id": employee["employee_id"],
        "employee_name": employee["name"],
        "days": days,
        "leave_type": leave_type,
        "reason": reason,
    }
    record_event("hr_leave_confirmation_requested", employee_id=employee["employee_id"], days=days)
    return _leave_confirmation_message(session["pending_leave"])


def respond(prompt: str) -> None:
    """Append a user turn, route it, and save the response to this session only."""
    session = active_session()
    user_message = HumanMessage(content=prompt)
    session["messages"].append(user_message)
    if session["title"] == "New conversation":
        session["title"] = _session_title(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding the right workplace specialist..."):
            with capture_activity() as activity:
                try:
                    record_event(
                        "chat_request_started",
                        session_id=session["id"],
                        message_count=len(session["messages"]),
                    )
                    leave_response = handle_hr_leave_request(session, prompt)
                    if leave_response is not None:
                        route = "hr_agent"
                        response = AIMessage(content=leave_response, additional_kwargs={"agent": "HR"})
                        st.caption("Routed to HR")
                        st.markdown(leave_response)
                        session["messages"].append(response)
                        record_event("chat_request_completed", session_id=session["id"], route=route)
                    else:
                        route = supervisor_node({"messages": session["messages"]})
                        result = workflow.invoke({"messages": session["messages"]})
                        responses = [message for message in result["messages"] if isinstance(message, AIMessage)]
                        if not responses:
                            record_event("chat_request_failed", reason="no_response")
                            st.warning("No response was generated. Please try again.")
                        else:
                            response = responses[-1]
                            response.additional_kwargs["agent"] = AGENT_LABELS.get(route, "Enterprise AI")
                            st.caption(f"Routed to {response.additional_kwargs['agent']}")
                            st.markdown(str(response.content))
                            session["messages"].append(response)
                            record_event("chat_request_completed", session_id=session["id"], route=route)
                except Exception as error:
                    record_event("chat_request_failed", session_id=session["id"], error=type(error).__name__)
                    st.error("I couldn't complete that request. Check that Ollama is running, then try again.")
                    with st.expander("Technical details"):
                        st.code(str(error))
            session["activity"].extend(activity)


initialize_sessions()
render_sidebar()

st.title("Enterprise AI")
st.caption("Ask naturally. Your message is automatically routed to the appropriate workplace specialist.")

messages = active_session()["messages"]
if not messages:
    st.info("Try asking about IT support, leave, expenses, travel, or workplace policies.")
render_messages(messages)

if prompt := st.chat_input("Message Enterprise AI"):
    respond(prompt)

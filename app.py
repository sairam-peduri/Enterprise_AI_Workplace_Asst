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
from src.tools.it_tools import (
    find_employee_in_text as it_find_employee_in_text,
    suggest_employee_in_text as it_suggest_employee_in_text,
    raise_it_ticket,
    reset_password,
    unlock_account,
)
from src.tools.finance_tools import (
    submit_expense,
    find_employee_in_text as finance_find_employee_in_text,
    suggest_employee_in_text as finance_suggest_employee_in_text,
)
from src.tools.travel_tools import (
    request_business_travel,
    cancel_travel_request,
    find_employee_in_text as travel_find_employee_in_text,
    suggest_employee_in_text as travel_suggest_employee_in_text,
)
from src.utils.logging_config import LOG_FILE, capture_activity, record_event
from src.utils.session_persistence import load_sessions, save_sessions


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
        "pending_leave_balance": None,
        "pending_it_ticket": None,
        "pending_it_action": None,
        "pending_expense": None,
        "pending_travel": None,
        "pending_reimbursement": None,
        "pending_budget": None,
        "pending_travel_plan": None,
    }


def initialize_sessions() -> None:
    """Initialize session-local conversations and select the first one."""
    if "chat_sessions" not in st.session_state:
        saved = load_sessions()
        if saved:
            for session in saved.values():
                session["pending_leave"] = None
                session["pending_leave_balance"] = None
                session["pending_it_ticket"] = None
                session["pending_it_action"] = None
                session["pending_expense"] = None
                session["pending_travel"] = None
                session["pending_reimbursement"] = None
                session["pending_budget"] = None
                session["pending_travel_plan"] = None
                session.setdefault("activity", [])
            st.session_state.chat_sessions = saved
            st.session_state.active_session_id = next(iter(saved))
        else:
            first_session = _new_session()
            st.session_state.chat_sessions = {first_session["id"]: first_session}
            st.session_state.active_session_id = first_session["id"]
    else:
        for session in st.session_state.chat_sessions.values():
            session.setdefault("activity", [])
            session.setdefault("pending_leave", None)
            session.setdefault("pending_leave_balance", None)
            session.setdefault("pending_it_ticket", None)
            session.setdefault("pending_it_action", None)
            session.setdefault("pending_expense", None)
            session.setdefault("pending_travel", None)
            session.setdefault("pending_reimbursement", None)
            session.setdefault("pending_budget", None)
            session.setdefault("pending_travel_plan", None)


def active_session() -> dict:
    return st.session_state.chat_sessions[st.session_state.active_session_id]


def _persist_sessions() -> None:
    """Save current sessions to disk."""
    save_sessions(st.session_state.chat_sessions)


def start_new_session() -> None:
    session = _new_session()
    st.session_state.chat_sessions[session["id"]] = session
    st.session_state.active_session_id = session["id"]
    _persist_sessions()


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
            created = datetime.fromisoformat(session["created_at"])
            time_label = created.strftime("%b %d, %H:%M")
            button_label = f"{session['title']}  \n*{time_label}*"
            if st.button(
                button_label,
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
            _persist_sessions()
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

    if pending and pending["stage"] == "awaiting_name":
        employee = find_employee_in_text(prompt)
        if employee is not None:
            session["pending_leave"] = {"stage": "collecting_details", "employee": employee}
            record_event("hr_employee_found_by_name", employee_id=employee["employee_id"])
            return (
                f"Employee found: {employee['name']} ({employee['employee_id']}). "
                "Please provide the number of days and a reason, for example: `5 days, reason: fever`."
            )
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

    if pending and pending["stage"] == "collecting_details":
        days, reason, leave_type = _leave_details(prompt)
        if days is None or reason is None:
            missing = []
            if days is None:
                missing.append("number of days")
            if reason is None:
                missing.append("reason")
            return f"Please provide the missing details: **{', '.join(missing)}**. For example: `5 days, reason: fever`."
        prepared = prepare_leave_application(pending["employee"]["name"], days, leave_type, reason)
        if not prepared["success"]:
            session["pending_leave"] = None
            record_event("hr_leave_validation_failed", employee_id=pending["employee"]["employee_id"])
            return prepared["message"]
        session["pending_leave"] = {
            "stage": "awaiting_confirmation",
            "employee_id": pending["employee"]["employee_id"],
            "employee_name": pending["employee"]["name"],
            "days": days,
            "leave_type": leave_type,
            "reason": reason,
        }
        record_event("hr_leave_confirmation_requested", employee_id=pending["employee"]["employee_id"], days=days)
        return _leave_confirmation_message(session["pending_leave"])

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

    # Only detect NEW leave application requests (exclude balance checks)
    is_leave_balance = any(w in normalized_prompt for w in (
        "check leave", "leave balance", "remaining leave", "leave remaining",
        "how many leave", "how many days leave",
    ))
    is_leave_request = (
        pending is not None or
        (not is_leave_balance and ("leave" in normalized_prompt or "time off" in normalized_prompt))
    )
    if not is_leave_request:
        return None

    employee = find_employee_in_text(prompt)
    if employee is not None:
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

    suggested_employee = suggest_employee_in_text(prompt)
    if suggested_employee is not None:
        session["pending_leave"] = {"stage": "awaiting_employee_confirmation", "employee": suggested_employee}
        record_event("hr_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
        return (
            f"Employee not found. Did you mean **{suggested_employee['name']}** "
            f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
        )

    record_event("hr_employee_not_found")
    session["pending_leave"] = {"stage": "awaiting_name"}
    return "I can help you apply for leave. Please provide your **full name** exactly as recorded in HR."


def handle_hr_leave_balance_check(session: dict, prompt: str) -> str | None:
    """Handle 'check leave balance' / 'remaining leaves' requests - just ask for name and show balance."""
    normalized_prompt = prompt.strip().casefold()
    pending = session.get("pending_leave_balance")

    is_check = any(w in normalized_prompt for w in (
        "check leave", "leave balance", "remaining leave", "leave remaining",
        "how many leave", "how many days leave", "balancing leave", "check remaining",
    ))
    if not is_check and pending is None:
        return None

    # Stage: Waiting for employee name
    if pending and pending["stage"] == "awaiting_name":
        employee = find_employee_in_text(prompt)
        if employee is None:
            employee = suggest_employee_in_text(prompt)
        if employee is None:
            return "Employee not found. Please provide your **full name** as recorded in the system."
        from src.tools.hr_tools import check_leave_balance
        result = check_leave_balance.invoke({"employee_id": employee["employee_id"]})
        session["pending_leave_balance"] = None
        record_event("hr_leave_balance_checked", employee_id=employee["employee_id"])
        return str(result)

    # Initial detection - ask for name
    session["pending_leave_balance"] = {"stage": "awaiting_name"}
    record_event("hr_leave_balance_check_started")
    return "I can help you check your leave balance. Please provide your **full name** as recorded in the system."


def _it_ticket_confirmation_message(ticket: dict) -> str:
    return (
        "Please confirm this IT ticket:\n\n"
        f"- Employee: {ticket['employee_name']} ({ticket['employee_id']})\n"
        f"- Issue: {ticket['issue']}\n\n"
        "Reply **yes** to submit the ticket, or **no** to cancel."
    )


def handle_it_ticket_request(session: dict, prompt: str) -> str | None:
    """Run the name-based, explicit-confirmation IT ticket flow for one session."""
    pending = session.get("pending_it_ticket")
    normalized_prompt = prompt.strip().casefold()

    # Stage: Waiting for user to provide their name
    if pending and pending["stage"] == "awaiting_name":
        employee = it_find_employee_in_text(prompt)
        if employee is not None:
            session["pending_it_ticket"] = {
                "stage": "collecting_reason",
                "employee_id": employee["employee_id"],
                "employee_name": employee["name"],
                "issue": pending.get("issue"),
            }
            record_event("it_employee_found_by_name", employee_id=employee["employee_id"])
            return (
                f"Employee found: {employee['name']} ({employee['employee_id']}). "
                "Please describe the reason for raising this IT ticket."
            )
        # Try fuzzy match
        suggested_employee = it_suggest_employee_in_text(prompt)
        if suggested_employee is not None:
            session["pending_it_ticket"] = {
                "stage": "awaiting_employee_confirmation",
                "employee": suggested_employee,
                "issue": pending.get("issue"),
            }
            record_event("it_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
            return (
                f"Did you mean **{suggested_employee['name']}** "
                f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
            )
        record_event("it_employee_not_found")
        return "Employee not found. Please provide your **full name** exactly as recorded in the system."

    # Stage: User confirmed suggested employee
    if pending and pending["stage"] == "awaiting_employee_confirmation":
        suggested_employee = pending["employee"]
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            session["pending_it_ticket"] = {
                "stage": "collecting_reason",
                "employee_id": suggested_employee["employee_id"],
                "employee_name": suggested_employee["name"],
                "issue": pending.get("issue"),
            }
            record_event("it_employee_suggestion_accepted", employee_id=suggested_employee["employee_id"])
            return (
                f"Employee confirmed: {suggested_employee['name']} ({suggested_employee['employee_id']}). "
                "Please describe the reason for raising this IT ticket."
            )
        if normalized_prompt in {"no", "n", "cancel"}:
            session["pending_it_ticket"] = None
            record_event("it_employee_suggestion_rejected")
            return "Employee not found. Please provide the employee's full name exactly as recorded in the system."
        return "Please reply **yes** to use the suggested employee or **no** to cancel."

    # Stage: Collecting reason from user
    if pending and pending["stage"] == "collecting_reason":
        issue = prompt.strip()
        if not issue:
            return "Please provide a reason for raising the IT ticket."
        session["pending_it_ticket"] = {
            "stage": "awaiting_confirmation",
            "employee_id": pending["employee_id"],
            "employee_name": pending["employee_name"],
            "issue": issue,
        }
        record_event("it_ticket_reason_collected", employee_id=pending["employee_id"])
        return _it_ticket_confirmation_message(session["pending_it_ticket"])

    # Stage: Awaiting final confirmation before raising ticket
    if pending and pending["stage"] == "awaiting_confirmation":
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            result = raise_it_ticket.invoke(
                {
                    "employee_id": pending["employee_id"],
                    "issue": pending["issue"],
                    "confirmed": True,
                }
            )
            session["pending_it_ticket"] = None
            result_text = str(result)
            record_event(
                "it_ticket_submitted" if "created successfully" in result_text else "it_ticket_submission_failed",
                employee_id=pending["employee_id"],
            )
            return result_text
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            session["pending_it_ticket"] = None
            record_event("it_ticket_cancelled", employee_id=pending["employee_id"])
            return "IT ticket cancelled. No data was changed."
        return "Please reply **yes** to submit the IT ticket or **no** to cancel it."

    # Detect if this is an IT request (but NOT password reset or unlock account - those are handled by handle_it_action_request)
    is_it_request = pending is not None or any(
        word in normalized_prompt for word in ("software install", "vpn", "hardware issue", "it ticket", "support ticket", "raise ticket", "create ticket", "new ticket", "raise a ticket", "raise an it ticket", "open ticket")
    )
    if not is_it_request:
        return None

    # If there's a stale pending_it_action from a different flow, clear it
    if pending is None and session.get("pending_it_action") is not None:
        session["pending_it_action"] = None

    # Extract issue type from the prompt
    issue = None
    if "software" in normalized_prompt and ("install" in normalized_prompt or "installation" in normalized_prompt):
        software_match = re.search(r"(?:install|installation)\s+(?:of\s+)?(.+?)(?:\s+for|\s*$)", prompt, flags=re.IGNORECASE)
        issue = f"Install {software_match.group(1).strip()}" if software_match else "Software installation request"
    elif "vpn" in normalized_prompt:
        issue = "VPN access request"
    elif "hardware" in normalized_prompt:
        issue_match = re.search(r"hardware\s+(?:issue|problem|error)\s*(?:with|:)?\s*(.+?)(?:\s+for|\s*$)", prompt, flags=re.IGNORECASE)
        issue = f"Hardware issue: {issue_match.group(1).strip()}" if issue_match else "Hardware issue"
    else:
        issue_match = re.search(r"(?:ticket|issue|problem)\s+(?:for|about|regarding)?\s*(.+?)(?:\s+for|\s*$)", prompt, flags=re.IGNORECASE)
        issue = issue_match.group(1).strip() if issue_match else "General IT issue"

    # Try to find employee in the current message
    employee = it_find_employee_in_text(prompt)
    if employee is not None:
        session["pending_it_ticket"] = {
            "stage": "awaiting_confirmation",
            "employee_id": employee["employee_id"],
            "employee_name": employee["name"],
            "issue": issue,
        }
        record_event("it_ticket_confirmation_requested", employee_id=employee["employee_id"])
        return _it_ticket_confirmation_message(session["pending_it_ticket"])

    # Try fuzzy match
    suggested_employee = it_suggest_employee_in_text(prompt)
    if suggested_employee is not None:
        session["pending_it_ticket"] = {
            "stage": "awaiting_employee_confirmation",
            "employee": suggested_employee,
            "issue": issue,
        }
        record_event("it_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
        return (
            f"Did you mean **{suggested_employee['name']}** "
            f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
        )

    # No employee found - ask for name first, but store the issue for later
    record_event("it_employee_not_found")
    session["pending_it_ticket"] = {
        "stage": "awaiting_name",
        "issue": issue,
    }
    return "I can help you raise an IT ticket. Please provide your **full name** as recorded in the system."


def _it_action_confirmation_message(action: dict) -> str:
    action_type = action.get("action_type", "IT action")
    if action_type == "reset_password":
        return (
            "Please confirm this password reset request:\n\n"
            f"- Employee: {action['employee_name']} ({action['employee_id']})\n"
            f"- Action: Password Reset\n"
            f"- A temporary password will be sent to the employee's email\n\n"
            "Reply **yes** to submit the request, or **no** to cancel."
        )
    elif action_type == "unlock_account":
        return (
            "Please confirm this account unlock request:\n\n"
            f"- Employee: {action['employee_name']} ({action['employee_id']})\n"
            f"- Action: Account Unlock\n\n"
            "Reply **yes** to submit the request, or **no** to cancel."
        )
    return "Please confirm this action. Reply **yes** to proceed or **no** to cancel."


def handle_it_action_request(session: dict, prompt: str) -> str | None:
    """Run the name-based, explicit-confirmation IT action flow (password reset, unlock account)."""
    pending = session.get("pending_it_action")
    normalized_prompt = prompt.strip().casefold()

    # If there's a stale pending ticket from a different flow, clear it
    if pending is None and session.get("pending_it_ticket") is not None:
        normalized_for_check = prompt.strip().casefold()
        is_it_action = any(
            word in normalized_for_check for word in ("forgot password", "reset password", "unlock account", "account locked")
        )
        if is_it_action:
            session["pending_it_ticket"] = None

    # Stage: Waiting for user to provide their name
    if pending and pending["stage"] == "awaiting_name":
        employee = it_find_employee_in_text(prompt)
        if employee is not None:
            session["pending_it_action"]["employee_id"] = employee["employee_id"]
            session["pending_it_action"]["employee_name"] = employee["name"]
            session["pending_it_action"]["stage"] = "awaiting_confirmation"
            record_event("it_action_employee_found_by_name", employee_id=employee["employee_id"])
            return _it_action_confirmation_message(session["pending_it_action"])
        # Try fuzzy match
        suggested_employee = it_suggest_employee_in_text(prompt)
        if suggested_employee is not None:
            session["pending_it_action"]["employee"] = suggested_employee
            session["pending_it_action"]["stage"] = "awaiting_employee_confirmation"
            record_event("it_action_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
            return (
                f"Did you mean **{suggested_employee['name']}** "
                f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
            )
        record_event("it_action_employee_not_found")
        return "Employee not found. Please provide your **full name** exactly as recorded in the system."

    # Stage: User confirmed suggested employee
    if pending and pending["stage"] == "awaiting_employee_confirmation":
        suggested_employee = pending.get("employee")
        if suggested_employee is None:
            session["pending_it_action"] = None
            return "An error occurred. Please start over."
        
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            session["pending_it_action"]["employee_id"] = suggested_employee["employee_id"]
            session["pending_it_action"]["employee_name"] = suggested_employee["name"]
            session["pending_it_action"]["stage"] = "awaiting_confirmation"
            record_event("it_action_employee_suggestion_accepted", employee_id=suggested_employee["employee_id"])
            return _it_action_confirmation_message(session["pending_it_action"])
        if normalized_prompt in {"no", "n", "cancel"}:
            session["pending_it_action"] = None
            record_event("it_action_employee_suggestion_rejected")
            return "Employee not found. Please provide the employee's full name exactly as recorded in the system."
        return "Please reply **yes** to use the suggested employee or **no** to cancel."

    # Stage: Awaiting final confirmation before executing IT action
    if pending and pending["stage"] == "awaiting_confirmation":
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            action_type = pending["action_type"]
            employee_id = pending["employee_id"]
            
            if action_type == "reset_password":
                result = reset_password.invoke({"employee_id": employee_id, "confirmed": True})
                session["pending_it_action"] = None
                record_event("it_password_reset_submitted", employee_id=employee_id)
                return result
            elif action_type == "unlock_account":
                result = unlock_account.invoke({"employee_id": employee_id, "confirmed": True})
                session["pending_it_action"] = None
                record_event("it_unlock_account_submitted", employee_id=employee_id)
                return result
            else:
                session["pending_it_action"] = None
                return "Unknown action type."
        
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            action_type = pending.get("action_type", "IT action")
            session["pending_it_action"] = None
            record_event("it_action_cancelled", action_type=action_type)
            return f"{action_type.replace('_', ' ').title()} cancelled. No data was changed."
        return "Please reply **yes** to submit the request or **no** to cancel it."

    # Detect if this is a password reset or unlock account request
    is_password_reset = "forgot password" in normalized_prompt or "reset password" in normalized_prompt
    is_unlock_account = "unlock account" in normalized_prompt or "account locked" in normalized_prompt
    
    if not is_password_reset and not is_unlock_account:
        return None

    # Try to find employee in the current message
    employee = it_find_employee_in_text(prompt)
    if employee is not None:
        action_type = "reset_password" if is_password_reset else "unlock_account"
        session["pending_it_action"] = {
            "stage": "awaiting_confirmation",
            "employee_id": employee["employee_id"],
            "employee_name": employee["name"],
            "action_type": action_type,
        }
        record_event("it_action_employee_found", employee_id=employee["employee_id"])
        return _it_action_confirmation_message(session["pending_it_action"])

    # Try fuzzy match
    suggested_employee = it_suggest_employee_in_text(prompt)
    if suggested_employee is not None:
        action_type = "reset_password" if is_password_reset else "unlock_account"
        session["pending_it_action"] = {
            "stage": "awaiting_employee_confirmation",
            "employee": suggested_employee,
            "action_type": action_type,
        }
        record_event("it_action_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
        return (
            f"Did you mean **{suggested_employee['name']}** "
            f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
        )

    # No employee found - ask for name first
    action_type = "reset_password" if is_password_reset else "unlock_account"
    record_event("it_action_employee_not_found")
    session["pending_it_action"] = {
        "stage": "awaiting_name",
        "action_type": action_type,
    }
    action_name = "password reset" if is_password_reset else "account unlock"
    return f"I can help you with {action_name}. Please provide your **full name** as recorded in the system."


def _reimbursement_status_message(expenses: list, employee_name: str, employee_id: str) -> str:
    """Format all expenses for an employee as a status summary."""
    if not expenses:
        return f"No expense claims found for {employee_name} ({employee_id})."
    lines = [f"Expense claims for **{employee_name}** ({employee_id}):\n"]
    for exp in expenses:
        lines.append(
            f"- **{exp['expense_id']}** | ₹{exp['amount']} | {exp['category']} | "
            f"{exp['description']} | {exp['expense_date']} | Status: {exp['status']}"
        )
    return "\n".join(lines)


def handle_finance_reimbursement_check(session: dict, prompt: str) -> str | None:
    """Handle 'check reimbursement status' requests - just ask for name, show all their expenses."""
    normalized_prompt = prompt.strip().casefold()
    pending = session.get("pending_reimbursement")

    # Check if this is a reimbursement status check request
    is_check = any(w in normalized_prompt for w in ("check reimbursement", "reimbursement status", "expense status", "my expenses", "my claims"))
    if not is_check and pending is None:
        return None

    # Stage: Waiting for employee name
    if pending and pending["stage"] == "awaiting_name":
        employee = finance_find_employee_in_text(prompt)
        if employee is None:
            employee = finance_suggest_employee_in_text(prompt)
        if employee is None:
            return "Employee not found. Please provide your **full name** as recorded in the system."
        # Look up all expenses for this employee
        from src.tools.finance_tools import load_expenses
        all_expenses = load_expenses()
        my_expenses = [e for e in all_expenses if e.get("employee_id", "").upper() == employee["employee_id"].upper()]
        session["pending_reimbursement"] = None
        record_event("finance_reimbursement_checked", employee_id=employee["employee_id"], count=len(my_expenses))
        return _reimbursement_status_message(my_expenses, employee["name"], employee["employee_id"])

    # Initial detection - ask for name
    session["pending_reimbursement"] = {"stage": "awaiting_name"}
    record_event("finance_reimbursement_check_started")
    return "I can help you check your reimbursement status. Please provide your **full name** as recorded in the system."


def _expense_confirmation_message(expense: dict) -> str:
    return (
        "Please confirm this expense claim:\n\n"
        f"- Employee: {expense['employee_name']} ({expense['employee_id']})\n"
        f"- Amount: ₹{expense['amount']}\n"
        f"- Category: {expense['category']}\n"
        f"- Description: {expense['description']}\n"
        f"- Expense Date: {expense['expense_date']}\n"
        f"- Receipt Available: {'Yes' if expense['receipt_available'] else 'No'}\n\n"
        "Reply **yes** to submit the expense claim, or **no** to cancel."
    )


def handle_finance_expense_request(session: dict, prompt: str) -> str | None:
    """Run the name-based, explicit-confirmation finance expense flow for one session."""
    pending = session.get("pending_expense")
    normalized_prompt = prompt.strip().casefold()

    # Stage: Waiting for user to provide their name
    if pending and pending["stage"] == "awaiting_name":
        employee = finance_find_employee_in_text(prompt)
        if employee is not None:
            session["pending_expense"] = {
                "stage": "collecting_details",
                "employee_id": employee["employee_id"],
                "employee_name": employee["name"],
            }
            record_event("finance_employee_found_by_name", employee_id=employee["employee_id"])
            return (
                f"Employee found: {employee['name']} ({employee['employee_id']}). "
                "Please provide the expense amount, category, description, date, and whether you have a receipt."
            )
        # Try fuzzy match
        suggested_employee = finance_suggest_employee_in_text(prompt)
        if suggested_employee is not None:
            session["pending_expense"] = {
                "stage": "awaiting_employee_confirmation",
                "employee": suggested_employee,
            }
            record_event("finance_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
            return (
                f"Did you mean **{suggested_employee['name']}** "
                f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
            )
        record_event("finance_employee_not_found")
        return "Employee not found. Please provide your **full name** exactly as recorded in the system."

    # Stage: User confirmed suggested employee
    if pending and pending["stage"] == "awaiting_employee_confirmation":
        suggested_employee = pending["employee"]
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            session["pending_expense"] = {
                "stage": "collecting_details",
                "employee_id": suggested_employee["employee_id"],
                "employee_name": suggested_employee["name"],
            }
            record_event("finance_employee_suggestion_accepted", employee_id=suggested_employee["employee_id"])
            return (
                f"Employee confirmed: {suggested_employee['name']} ({suggested_employee['employee_id']}). "
                "Please provide the expense amount, category, description, date, and whether you have a receipt."
            )
        if normalized_prompt in {"no", "n", "cancel"}:
            session["pending_expense"] = None
            record_event("finance_employee_suggestion_rejected")
            return "Employee not found. Please provide the employee's full name exactly as recorded in the system."
        return "Please reply **yes** to use the suggested employee or **no** to cancel."

    # Stage: Collecting expense details from user
    if pending and pending["stage"] == "collecting_details":
        # Handle cancel/no at any point
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            session["pending_expense"] = None
            record_event("finance_expense_cancelled", employee_id=pending["employee_id"])
            return "Expense claim cancelled. No data was changed."

        # Amount: look for numbers with optional currency symbols
        amount = None
        amount_match = re.search(r"(?:amount|cost|total|price)\s*(?:is|:|=)?\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)", prompt, flags=re.IGNORECASE)
        if amount_match:
            amount = float(amount_match.group(1))
        else:
            amount_match = re.search(r"(?:₹|rs\.?|inr)\s*(\d+(?:\.\d+)?)", prompt, flags=re.IGNORECASE)
            if amount_match:
                amount = float(amount_match.group(1))
            else:
                amount_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:rs|inr|rupees|/-)\b", prompt, flags=re.IGNORECASE)
                if amount_match:
                    amount = float(amount_match.group(1))
                else:
                    # Try standalone number at start or after comma
                    amount_match = re.search(r"(?:^|,)\s*(?:amount\s*(?:is|:|=)?\s*)?(\d+(?:\.\d+)?)", prompt, flags=re.IGNORECASE)
                    if amount_match:
                        amount = float(amount_match.group(1))

        # Category
        category = None
        category_match = re.search(r"\b(category|type)\s*(?:is|:|=)?\s*(.+?)(?:\s*,|\s+and|\s+for|\s+with|\s*$)", prompt, flags=re.IGNORECASE)
        if category_match:
            category = category_match.group(2).strip()
        else:
            known_categories = ["international travel", "local travel", "travel", "food", "accommodation", "office supplies", "software", "hardware", "training", "medical", "other"]
            for cat in known_categories:
                if cat in normalized_prompt:
                    category = cat.title()
                    break

        # Description
        description = None
        desc_match = re.search(r"\b(description|desc|details|purpose)\s*(?:is|:|=)?\s*(.+?)(?:\s*,|\s+and|\s+for|\s+on|\s*$)", prompt, flags=re.IGNORECASE)
        if desc_match:
            description = desc_match.group(2).strip()
        else:
            desc_match = re.search(r"\b(?:for|about|regarding)\s+(.+?)(?:\s*,|\s+on|\s+for|\s*$)", prompt, flags=re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1).strip()

        # Date - support DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, "28 July 2026"
        expense_date = None
        date_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", prompt)
        if date_match:
            d, m, y = date_match.group(1), date_match.group(2), date_match.group(3)
            expense_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        else:
            date_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", prompt)
            if date_match:
                y, m, d = date_match.group(1), date_match.group(2), date_match.group(3)
                expense_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            else:
                date_match = re.search(r"\b(\d{1,2})\s+(\w+)\s+(\d{4})\b", prompt)
                if date_match:
                    expense_date = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1).zfill(2)}"

        # Receipt
        has_receipt = True
        if re.search(r"\b(no\s+receipt|without\s+receipt)\b", normalized_prompt):
            has_receipt = False
        elif re.search(r"\b(receipt|invoice)\s*(?:is|:|=)?\s*(yes|no|available|not available|attached|not attached)\b", prompt, flags=re.IGNORECASE):
            receipt_val = re.search(r"\b(receipt|invoice)\s*(?:is|:|=)?\s*(yes|no|available|not available|attached|not attached)\b", prompt, flags=re.IGNORECASE)
            has_receipt = receipt_val.group(2).lower() in {"yes", "available", "attached"}
        elif re.search(r"\b(have|i have|with|had)\s+(a\s+)?receipt\b", normalized_prompt):
            has_receipt = True
        elif re.search(r"\b(don.t have|do not have|no)\s+(a\s+)?receipt\b", normalized_prompt):
            has_receipt = False

        # Check what's missing and ask for it one at a time
        missing = []
        if amount is None:
            missing.append("amount")
        if category is None:
            missing.append("category")
        if description is None:
            missing.append("description")
        if expense_date is None:
            missing.append("date")

        if missing:
            return (
                f"I have the employee: {pending['employee_name']} ({pending['employee_id']}). "
                f"Please provide the missing details: **{', '.join(missing)}**.\n"
                "For example: `amount: 5000, category: travel, description: client meeting, date: 2026-01-15, receipt: yes`"
            )

        session["pending_expense"] = {
            "stage": "awaiting_confirmation",
            "employee_id": pending["employee_id"],
            "employee_name": pending["employee_name"],
            "amount": amount,
            "category": category,
            "description": description,
            "expense_date": expense_date,
            "receipt_available": has_receipt,
        }
        record_event("finance_expense_details_collected", employee_id=pending["employee_id"])
        return _expense_confirmation_message(session["pending_expense"])

    # Stage: Awaiting final confirmation before submitting expense
    if pending and pending["stage"] == "awaiting_confirmation":
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            result = submit_expense(
                employee_id=pending["employee_id"],
                amount=pending["amount"],
                category=pending["category"],
                description=pending["description"],
                expense_date=pending["expense_date"],
                receipt_available=pending["receipt_available"],
                confirmed=True,
            )
            session["pending_expense"] = None
            if result["success"]:
                record_event("finance_expense_submitted", employee_id=pending["employee_id"])
                expense = result.get("expense", {})
                return (
                    f"Expense claim {expense.get('expense_id', 'N/A')} has been submitted successfully.\n"
                    f"Status: {expense.get('status', 'N/A')}"
                )
            else:
                record_event("finance_expense_submission_failed", employee_id=pending["employee_id"])
                return result["message"]
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            session["pending_expense"] = None
            record_event("finance_expense_cancelled", employee_id=pending["employee_id"])
            return "Expense claim cancelled. No data was changed."
        return "Please reply **yes** to submit the expense claim or **no** to cancel it."

    # Detect if this is a finance expense request (but NOT reimbursement check)
    is_check_reimbursement = any(w in normalized_prompt for w in ("check reimbursement", "reimbursement status", "expense status", "my expenses", "my claims"))
    is_expense_request = pending is not None or (
        not is_check_reimbursement and
        not session.get("pending_leave") and
        not session.get("pending_it_ticket") and
        not session.get("pending_it_action") and
        not session.get("pending_travel") and
        any(word in normalized_prompt for word in ("expense", "reimbursement", "claim", "receipt", "reimburse", "submit expense", "file expense"))
    )
    if not is_expense_request:
        return None

    # Try to find employee in the current message
    employee = finance_find_employee_in_text(prompt)
    if employee is not None:
        session["pending_expense"] = {
            "stage": "collecting_details",
            "employee_id": employee["employee_id"],
            "employee_name": employee["name"],
        }
        record_event("finance_employee_found", employee_id=employee["employee_id"])
        return (
            f"Employee found: {employee['name']} ({employee['employee_id']}). "
            "Please provide the expense amount, category, description, date, and whether you have a receipt."
        )

    # Try fuzzy match
    suggested_employee = finance_suggest_employee_in_text(prompt)
    if suggested_employee is not None:
        session["pending_expense"] = {
            "stage": "awaiting_employee_confirmation",
            "employee": suggested_employee,
        }
        record_event("finance_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
        return (
            f"Did you mean **{suggested_employee['name']}** "
            f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
        )

    # No employee found - ask for name first
    record_event("finance_employee_not_found")
    session["pending_expense"] = {
        "stage": "awaiting_name",
    }
    return "I can help you submit an expense claim. Please provide your **full name** as recorded in the system."


def _travel_confirmation_message(travel: dict) -> str:
    return (
        "Please confirm this travel request:\n\n"
        f"- Employee: {travel['employee_name']} ({travel['employee_id']})\n"
        f"- From: {travel['source']}\n"
        f"- To: {travel['destination']}\n"
        f"- Dates: {travel['start_date']} to {travel['end_date']}\n"
        f"- Purpose: {travel['purpose']}\n\n"
        "Reply **yes** to submit the travel request, or **no** to cancel."
    )


def _travel_cancel_confirmation_message(travel: dict) -> str:
    return (
        "Please confirm cancellation of this travel request:\n\n"
        f"- Request ID: {travel['request_id']}\n"
        f"- Employee: {travel['employee_name']} ({travel['employee_id']})\n"
        f"- Destination: {travel['destination']}\n"
        f"- Dates: {travel['start_date']} to {travel['end_date']}\n\n"
        "Reply **yes** to cancel the travel request, or **no** to keep it."
    )


def handle_estimate_budget(session: dict, prompt: str) -> str | None:
    """Handle 'estimate budget' requests - ask for destination and days, call tool."""
    normalized_prompt = prompt.strip().casefold()
    pending = session.get("pending_budget")

    is_budget = any(w in normalized_prompt for w in ("estimate budget", "budget estimate", "travel cost", "trip cost", "how much does travel cost"))
    if not is_budget and pending is None:
        return None

    # Stage: Collecting destination and days
    if pending and pending["stage"] == "collecting":
        if normalized_prompt in {"no", "n", "cancel"}:
            session["pending_budget"] = None
            return "Budget estimate cancelled."

        dest_match = re.search(r"(?:to|for|destination)\s*(?:is|:|=)?\s*(\w[\w\s]*?)(?:\s*,|\s+for|\s+days|\s*$)", prompt, flags=re.IGNORECASE)
        days_match = re.search(r"\b(\d+)\s*days?\b", prompt, flags=re.IGNORECASE)

        destination = None
        if dest_match:
            destination = dest_match.group(1).strip()
        else:
            parts = [p.strip() for p in prompt.split(",")]
            for part in parts:
                cleaned = re.sub(r"^(?:to|for|destination)\s*(?:is|:|=)?\s*", "", part, flags=re.IGNORECASE).strip()
                if cleaned and not re.match(r"^\d+\s*days?$", cleaned, flags=re.IGNORECASE):
                    destination = cleaned
                    break

        days = int(days_match.group(1)) if days_match else None

        if not destination or not days:
            missing = []
            if not destination: missing.append("destination")
            if not days: missing.append("number of days")
            return f"Please provide the missing details: **{', '.join(missing)}**.\nExample: `Delhi, 3 days`"

        session["pending_budget"] = None
        from src.tools.travel_tools import estimate_budget
        result = estimate_budget.invoke({"destination": destination, "days": days})
        record_event("travel_budget_estimated", destination=destination, days=days)
        return str(result)

    # Initial detection - ask for destination and days
    session["pending_budget"] = {"stage": "collecting"}
    record_event("travel_budget_request_started")
    return "I can estimate your travel budget. Please provide the **destination** and **number of days**, for example: `Delhi, 3 days`"


def handle_generate_travel_plan(session: dict, prompt: str) -> str | None:
    """Handle 'generate travel plan' requests - ask for destination, days, purpose."""
    normalized_prompt = prompt.strip().casefold()
    pending = session.get("pending_travel_plan")

    is_plan = any(w in normalized_prompt for w in ("generate travel plan", "create travel plan", "travel itinerary", "plan my trip", "travel plan"))
    if not is_plan and pending is None:
        return None

    # Stage: Collecting details
    if pending and pending["stage"] == "collecting":
        if normalized_prompt in {"no", "n", "cancel"}:
            session["pending_travel_plan"] = None
            return "Travel plan generation cancelled."

        dest_match = re.search(r"(?:to|for|destination)\s*(?:is|:|=)?\s*(\w[\w\s]*?)(?:\s*,|\s+for|\s+days|\s*$)", prompt, flags=re.IGNORECASE)
        days_match = re.search(r"\b(\d+)\s*days?\b", prompt, flags=re.IGNORECASE)
        purpose_match = re.search(r"\b(?:purpose|reason|for)\s*(?:is|:|=)?\s*(.+?)(?:\s*,|\s*$)", prompt, flags=re.IGNORECASE)

        destination = None
        if dest_match:
            destination = dest_match.group(1).strip()
        else:
            parts = [p.strip() for p in prompt.split(",")]
            for part in parts:
                cleaned = re.sub(r"^(?:to|for|destination)\s*(?:is|:|=)?\s*", "", part, flags=re.IGNORECASE).strip()
                if cleaned and not re.match(r"^\d+\s*days?$", cleaned, flags=re.IGNORECASE) and not re.match(r"\d{4}-\d{2}-\d{2}", cleaned):
                    destination = cleaned
                    break

        days = int(days_match.group(1)) if days_match else None

        purpose = None
        if purpose_match:
            purpose = purpose_match.group(1).strip()
        else:
            for purp in ["client meeting", "training", "conference", "customer visit", "audit", "business meeting"]:
                if purp in normalized_prompt:
                    purpose = purp
                    break
        if not purpose:
            purpose = "Client Meeting"

        if not destination or not days:
            missing = []
            if not destination: missing.append("destination")
            if not days: missing.append("number of days")
            return f"Please provide the missing details: **{', '.join(missing)}**.\nExample: `Delhi, 3 days, client meeting`"

        session["pending_travel_plan"] = None
        from src.tools.travel_tools import generate_travel_plan
        result = generate_travel_plan.invoke({"destination": destination, "days": days, "purpose": purpose})
        record_event("travel_plan_generated", destination=destination, days=days, purpose=purpose)
        return str(result)

    # Initial detection - ask for details
    session["pending_travel_plan"] = {"stage": "collecting"}
    record_event("travel_plan_request_started")
    return "I can generate a travel plan for you. Please provide the **destination**, **number of days**, and **purpose**, for example: `Delhi, 3 days, client meeting`"


def handle_travel_request(session: dict, prompt: str) -> str | None:
    """Run the name-based, explicit-confirmation travel flow for one session."""
    pending = session.get("pending_travel")
    normalized_prompt = prompt.strip().casefold()

    # Stage: Waiting for user to provide their name
    if pending and pending["stage"] == "awaiting_name":
        employee = travel_find_employee_in_text(prompt)
        if employee is not None:
            session["pending_travel"]["employee_id"] = employee["employee_id"]
            session["pending_travel"]["employee_name"] = employee["name"]
            session["pending_travel"]["stage"] = "collecting_details"
            record_event("travel_employee_found_by_name", employee_id=employee["employee_id"])
            
            if pending.get("action") == "cancel":
                return (
                    f"Employee found: {employee['name']} ({employee['employee_id']}). "
                    "Please provide the travel request ID you want to cancel."
                )
            else:
                return (
                    f"Employee found: {employee['name']} ({employee['employee_id']}). "
                    "Please provide the source, destination, travel dates, and purpose."
                )
        # Try fuzzy match
        suggested_employee = travel_suggest_employee_in_text(prompt)
        if suggested_employee is not None:
            session["pending_travel"]["employee"] = suggested_employee
            session["pending_travel"]["stage"] = "awaiting_employee_confirmation"
            record_event("travel_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
            return (
                f"Did you mean **{suggested_employee['name']}** "
                f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
            )
        record_event("travel_employee_not_found")
        return "Employee not found. Please provide your **full name** exactly as recorded in the system."

    # Stage: User confirmed suggested employee
    if pending and pending["stage"] == "awaiting_employee_confirmation":
        suggested_employee = pending.get("employee")
        if suggested_employee is None:
            session["pending_travel"] = None
            return "An error occurred. Please start over."
        
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            session["pending_travel"]["employee_id"] = suggested_employee["employee_id"]
            session["pending_travel"]["employee_name"] = suggested_employee["name"]
            session["pending_travel"]["stage"] = "collecting_details"
            record_event("travel_employee_suggestion_accepted", employee_id=suggested_employee["employee_id"])
            
            if pending.get("action") == "cancel":
                return (
                    f"Employee confirmed: {suggested_employee['name']} ({suggested_employee['employee_id']}). "
                    "Please provide the travel request ID you want to cancel."
                )
            else:
                return (
                    f"Employee confirmed: {suggested_employee['name']} ({suggested_employee['employee_id']}). "
                    "Please provide the source, destination, travel dates, and purpose."
                )
        if normalized_prompt in {"no", "n", "cancel"}:
            session["pending_travel"] = None
            record_event("travel_employee_suggestion_rejected")
            return "Employee not found. Please provide the employee's full name exactly as recorded in the system."
        return "Please reply **yes** to use the suggested employee or **no** to cancel."

    # Stage: Collecting travel request ID for cancellation
    if pending and pending["stage"] == "collecting_request_id":
        request_id_match = re.search(r"\b(TR\d+)\b", prompt, flags=re.IGNORECASE)
        if request_id_match:
            request_id = request_id_match.group(1).upper()
            session["pending_travel"]["request_id"] = request_id
            session["pending_travel"]["stage"] = "awaiting_cancellation_confirmation"
            record_event("travel_request_id_collected", request_id=request_id)
            return _travel_cancel_confirmation_message(session["pending_travel"])
        return "Please provide a valid travel request ID (e.g., TR001)."

    # Stage: Awaiting final confirmation before cancelling travel
    if pending and pending["stage"] == "awaiting_cancellation_confirmation":
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            result = cancel_travel_request.invoke({
                "request_id": pending["request_id"],
                "confirmed": True,
            })
            session["pending_travel"] = None
            result_text = str(result)
            record_event(
                "travel_cancelled" if "cancelled successfully" in result_text else "travel_cancellation_failed",
                request_id=pending["request_id"],
            )
            return result_text
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            session["pending_travel"] = None
            record_event("travel_cancellation_aborted", request_id=pending["request_id"])
            return "Travel request cancellation aborted. No data was changed."
        return "Please reply **yes** to cancel the travel request or **no** to keep it."

    # Stage: Collecting travel details from user
    if pending and pending["stage"] == "collecting_details":
        # Handle cancel
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            session["pending_travel"] = None
            record_event("travel_cancelled_by_user", employee_id=pending.get("employee_id"))
            return "Travel request cancelled. No data was changed."

        source = None
        destination = None
        start_date = None
        end_date = None
        purpose = None

        parts = [p.strip() for p in prompt.split(",")]

        # Try labeled format: "source:Hyderabad, destination:delhi, start_date:2026-05-05, end_date:2026-06-06, purpose:cjp protest"
        src_label = re.search(r"\b(?:from|source|origin)\s*(?:is|:|=)?\s*(.+?)(?:\s*,\s*$|\s*$)", prompt, flags=re.IGNORECASE)
        dst_label = re.search(r"\b(?:to|destination|going to|visiting)\s*(?:is|:|=)?\s*(.+?)(?:\s*,\s*$|\s*$)", prompt, flags=re.IGNORECASE)
        purpose_label = re.search(r"\b(?:purpose|reason)\s*(?:is|:|=)?\s*(.+?)(?:\s*,\s*$|\s*$)", prompt, flags=re.IGNORECASE)

        # Extract dates (any format YYYY-MM-DD or DD-MM-YYYY)
        all_dates = re.findall(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", prompt)
        all_dates_iso = re.findall(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", prompt)

        # Parse dates to YYYY-MM-DD
        parsed_dates = []
        for d, m, y in all_dates:
            parsed_dates.append(f"{y}-{m.zfill(2)}-{d.zfill(2)}")
        for y, m, d in all_dates_iso:
            parsed_dates.append(f"{y}-{m.zfill(2)}-{d.zfill(2)}")

        if len(parsed_dates) >= 2:
            start_date = parsed_dates[0]
            end_date = parsed_dates[1]

        # Source
        if src_label:
            source = src_label.group(1).strip().rstrip(",")
        elif len(parts) >= 1:
            candidate = parts[0].strip()
            candidate = re.sub(r"^(?:from|source|origin)\s*(?:is|:|=)?\s*", "", candidate, flags=re.IGNORECASE).strip()
            if candidate and not re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", candidate):
                source = candidate

        # Destination
        if dst_label:
            destination = dst_label.group(1).strip().rstrip(",")
        elif len(parts) >= 2:
            candidate = parts[1].strip()
            candidate = re.sub(r"^(?:to|destination|going to|visiting)\s*(?:is|:|=)?\s*", "", candidate, flags=re.IGNORECASE).strip()
            if candidate and not re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", candidate):
                destination = candidate

        # Purpose
        if purpose_label:
            purpose = purpose_label.group(1).strip().rstrip(",")
        elif len(parts) >= 5:
            purpose = parts[-1].strip()
        elif len(parts) >= 4:
            last = parts[-1].strip()
            if not re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", last):
                purpose = last

        # Check known purpose keywords
        if not purpose:
            for purp in ["client meeting", "training", "conference", "customer visit", "audit", "business meeting", "cjp protest", "protest"]:
                if purp in normalized_prompt:
                    purpose = purp
                    break

        if source is None or destination is None or start_date is None or end_date is None or purpose is None:
            missing = []
            if source is None: missing.append("source")
            if destination is None: missing.append("destination")
            if start_date is None: missing.append("start date")
            if end_date is None: missing.append("end date")
            if purpose is None: missing.append("purpose")
            return (
                f"I have the employee: {pending['employee_name']} ({pending['employee_id']}). "
                f"Please provide the missing details: **{', '.join(missing)}**.\n"
                "Format: `source, destination, start_date, end_date, purpose`\n"
                "Example: `Mumbai, Delhi, 2026-01-15, 2026-01-18, client meeting`"
            )

        session["pending_travel"] = {
            "stage": "awaiting_confirmation",
            "employee_id": pending["employee_id"],
            "employee_name": pending["employee_name"],
            "source": source,
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "purpose": purpose,
        }
        record_event("travel_details_collected", employee_id=pending["employee_id"])
        return _travel_confirmation_message(session["pending_travel"])

    # Stage: Awaiting final confirmation before submitting travel request
    if pending and pending["stage"] == "awaiting_confirmation":
        if normalized_prompt in {"yes", "y", "confirm", "confirm yes"}:
            result = request_business_travel.invoke({
                "employee_id": pending["employee_id"],
                "source": pending["source"],
                "destination": pending["destination"],
                "start_date": pending["start_date"],
                "end_date": pending["end_date"],
                "purpose": pending["purpose"],
                "confirmed": True,
            })
            session["pending_travel"] = None
            result_text = str(result)
            record_event(
                "travel_submitted" if "submitted successfully" in result_text else "travel_submission_failed",
                employee_id=pending["employee_id"],
            )
            return result_text
        if normalized_prompt in {"no", "n", "cancel", "cancel request"}:
            session["pending_travel"] = None
            record_event("travel_cancelled_by_user", employee_id=pending["employee_id"])
            return "Travel request cancelled. No data was changed."
        return "Please reply **yes** to submit the travel request or **no** to cancel it."

    # Detect if this is a travel request (only if no other flow is active)
    is_travel_request = pending is not None or (
        not session.get("pending_leave") and
        not session.get("pending_leave_balance") and
        not session.get("pending_it_ticket") and
        not session.get("pending_it_action") and
        not session.get("pending_expense") and
        not session.get("pending_reimbursement") and
        any(word in normalized_prompt for word in ("travel", "trip", "flight", "hotel", "booking", "itinerary", "visit", "conference", "request business travel"))
    )
    is_cancel_request = "cancel" in normalized_prompt and any(
        word in normalized_prompt for word in ("travel", "trip", "booking", "request")
    )
    
    if not is_travel_request:
        return None

    # Try to find employee in the current message
    employee = travel_find_employee_in_text(prompt)
    if employee is not None:
        if is_cancel_request:
            session["pending_travel"] = {
                "stage": "collecting_request_id",
                "employee_id": employee["employee_id"],
                "employee_name": employee["name"],
                "action": "cancel",
            }
            record_event("travel_employee_found", employee_id=employee["employee_id"])
            return (
                f"Employee found: {employee['name']} ({employee['employee_id']}). "
                "Please provide the travel request ID you want to cancel (e.g., TR001)."
            )
        else:
            session["pending_travel"] = {
                "stage": "collecting_details",
                "employee_id": employee["employee_id"],
                "employee_name": employee["name"],
            }
            record_event("travel_employee_found", employee_id=employee["employee_id"])
            return (
                f"Employee found: {employee['name']} ({employee['employee_id']}). "
                "Please provide the source, destination, travel dates, and purpose."
            )

    # Try fuzzy match
    suggested_employee = travel_suggest_employee_in_text(prompt)
    if suggested_employee is not None:
        session["pending_travel"] = {
            "stage": "awaiting_employee_confirmation",
            "employee": suggested_employee,
            "action": "cancel" if is_cancel_request else "submit",
        }
        record_event("travel_employee_suggestion_requested", employee_id=suggested_employee["employee_id"])
        return (
            f"Did you mean **{suggested_employee['name']}** "
            f"({suggested_employee['employee_id']})? Reply **yes** to use this employee or **no** to cancel."
        )

    # No employee found - ask for name first
    record_event("travel_employee_not_found")
    session["pending_travel"] = {
        "stage": "awaiting_name",
        "action": "cancel" if is_cancel_request else "submit",
    }
    return "I can help you with travel requests. Please provide your **full name** as recorded in the system."


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
                    leave_balance_response = handle_hr_leave_balance_check(session, prompt)
                    if leave_balance_response is not None:
                        route = "hr_agent"
                        response = AIMessage(content=leave_balance_response, additional_kwargs={"agent": "HR"})
                        st.caption("Routed to HR")
                        st.markdown(leave_balance_response)
                        session["messages"].append(response)
                        record_event("chat_request_completed", session_id=session["id"], route=route)
                    else:
                        leave_response = handle_hr_leave_request(session, prompt)
                        if leave_response is not None:
                            route = "hr_agent"
                            response = AIMessage(content=leave_response, additional_kwargs={"agent": "HR"})
                            st.caption("Routed to HR")
                            st.markdown(leave_response)
                            session["messages"].append(response)
                            record_event("chat_request_completed", session_id=session["id"], route=route)
                        else:
                            it_action_response = handle_it_action_request(session, prompt)
                            if it_action_response is not None:
                                route = "it_agent"
                                response = AIMessage(content=it_action_response, additional_kwargs={"agent": "IT Support"})
                                st.caption("Routed to IT Support")
                                st.markdown(it_action_response)
                                session["messages"].append(response)
                                record_event("chat_request_completed", session_id=session["id"], route=route)
                            else:
                                it_response = handle_it_ticket_request(session, prompt)
                                if it_response is not None:
                                    route = "it_agent"
                                    response = AIMessage(content=it_response, additional_kwargs={"agent": "IT Support"})
                                    st.caption("Routed to IT Support")
                                    st.markdown(it_response)
                                    session["messages"].append(response)
                                    record_event("chat_request_completed", session_id=session["id"], route=route)
                                else:
                                    reimb_response = handle_finance_reimbursement_check(session, prompt)
                                    if reimb_response is not None:
                                        route = "finance_agent"
                                        response = AIMessage(content=reimb_response, additional_kwargs={"agent": "Finance"})
                                        st.caption("Routed to Finance")
                                        st.markdown(reimb_response)
                                        session["messages"].append(response)
                                        record_event("chat_request_completed", session_id=session["id"], route=route)
                                    else:
                                        finance_response = handle_finance_expense_request(session, prompt)
                                        if finance_response is not None:
                                            route = "finance_agent"
                                            response = AIMessage(content=finance_response, additional_kwargs={"agent": "Finance"})
                                            st.caption("Routed to Finance")
                                            st.markdown(finance_response)
                                            session["messages"].append(response)
                                            record_event("chat_request_completed", session_id=session["id"], route=route)
                                        else:
                                            budget_response = handle_estimate_budget(session, prompt)
                                            if budget_response is not None:
                                                route = "travel_agent"
                                                response = AIMessage(content=budget_response, additional_kwargs={"agent": "Travel"})
                                                st.caption("Routed to Travel")
                                                st.markdown(budget_response)
                                                session["messages"].append(response)
                                                record_event("chat_request_completed", session_id=session["id"], route=route)
                                            else:
                                                plan_response = handle_generate_travel_plan(session, prompt)
                                                if plan_response is not None:
                                                    route = "travel_agent"
                                                    response = AIMessage(content=plan_response, additional_kwargs={"agent": "Travel"})
                                                    st.caption("Routed to Travel")
                                                    st.markdown(plan_response)
                                                    session["messages"].append(response)
                                                    record_event("chat_request_completed", session_id=session["id"], route=route)
                                                else:
                                                    travel_response = handle_travel_request(session, prompt)
                                                    if travel_response is not None:
                                                        route = "travel_agent"
                                                        response = AIMessage(content=travel_response, additional_kwargs={"agent": "Travel"})
                                                        st.caption("Routed to Travel")
                                                        st.markdown(travel_response)
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
            _persist_sessions()


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

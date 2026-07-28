import re
from difflib import SequenceMatcher

from langchain_core.tools import tool 
from src.utils.file_handler import load_json,save_json 
from src.utils.paths import (EMPLOYEE_FILE,TICKET_FILE,SYSTEM_FILE)

"""
IT Tools Module

This module contains IT support tools used by the IT Agent.
These tools simulate common enterprise IT support operations.
"""
def _load_employees():
    return load_json(EMPLOYEE_FILE)

def _load_tickets():
    return load_json(TICKET_FILE)

def _save_tickets(tickets):
    save_json(TICKET_FILE,tickets) 

def _load_systems():
    return load_json(SYSTEM_FILE)

def _find_employee(employee_id,employees):
    employee_id = employee_id.strip().upper()
    return next(
        (emp for emp in employees if emp.get("employee_id") == employee_id),
        None,
    )

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


def _generate_ticket_id(tickets):
    """
    Generates ticket IDs like:
    IT001
    IT002
    IT003
    """

    if not tickets:
        return "IT001"
    numbers=[]
    for ticket in tickets:
        try:
            numbers.append(int(ticket.get("ticket_id", "").replace("IT", "")))
        except ValueError:
            continue
    next_number=max(numbers)+1 
    return f"IT{next_number:03d}"

@tool 
def reset_password(employee_id:str, confirmed: bool = False)->str:
    """
    Reset the password for a specific employee. Only resets when confirmed is True.

    Use ONLY when the user explicitly requests a password reset
    and provides a valid employee ID such as EMP001.

    Do NOT use this tool to answer questions about
    how password resets work.
    """

    if not confirmed:
        return "Confirmation required. Show the password reset summary and ask the employee to reply yes before submitting."

    employees=_load_employees()

    employee=_find_employee(employee_id,employees)

    if employee is None:
        return (
    f"No employee found with Employee ID '{employee_id}'. "
    "Please verify the Employee ID and try again."
)
    
    return (
        f"Password reset request has been successfully processed. "
        f"{employee['name']} ({employee_id}). "
        f"A temporary password has been sent to {employee['email']}."
    )

@tool
def unlock_account(employee_id:str, confirmed: bool = False)->str:
    """
    Unlock a locked employee account. Only unlocks when confirmed is True.
    
    Args:
        employee_id:Employee ID 
        confirmed: Whether the user has confirmed the action
        
    Returns:
        Success message.
    """

    if not confirmed:
        return "Confirmation required. Show the account unlock summary and ask the employee to reply yes before submitting."

    employees = _load_employees()

    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    if not employee.get("account_locked", False):
        return "Employee account is already unlocked."

    employee["account_locked"] = False

    save_json(EMPLOYEE_FILE, employees)

    return f"{employee['name']}'s account has been unlocked."

@tool
def raise_it_ticket(employee_id: str, issue: str, confirmed: bool = False) -> str:
    """
    Raise an IT support ticket.

    This tool changes mock IT data only when confirmed is True.
    """

    if not confirmed:
        return "Confirmation required. Show the ticket summary and ask the employee to reply yes before submitting."

    employees = _load_employees()

    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    tickets = _load_tickets()

    ticket_id = _generate_ticket_id(tickets)

    ticket = {
        "ticket_id": ticket_id,
        "employee_id": employee_id,
        "employee_name": employee["name"],
        "issue": issue,
        "status": "Open",
    }

    tickets.append(ticket)
    _save_tickets(tickets)

    return (
        f"Ticket {ticket_id} has been created successfully "
        f"for {employee['name']}."
    )

@tool
def check_ticket_status(ticket_id: str) -> str:
    """
    Check status of an IT ticket.
    """

    tickets = _load_tickets()

    ticket = next(
        (t for t in tickets if t["ticket_id"] == ticket_id),
        None,
    )

    if ticket is None:
        return "Ticket not found."

    return (
        f"Ticket {ticket_id}\n"
        f"Issue : {ticket['issue']}\n"
        f"Status : {ticket['status']}"
    )

@tool
def request_software_installation(
    employee_id: str,
    software_name: str,
) -> str:
    """
    Raise request for software installation.
    """

    employees = _load_employees()

    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    tickets = _load_tickets()

    ticket_id = _generate_ticket_id(tickets)


    tickets.append(
        {
            "ticket_id": ticket_id,
            "employee_id": employee_id,
            "employee_name": employee["name"],
            "issue": f"Install {software_name}",
            "status": "Open",
        }
    )

    _save_tickets(tickets)

    return (
        f"Software installation request submitted.\n"
        f"Ticket ID : {ticket_id}"
    )

@tool
def request_vpn_access(employee_id: str) -> str:
    """
    Raise VPN access request.
    """

    employees = _load_employees()

    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    return (
        f"VPN access request submitted successfully "
        f"for {employee['name']}."
    )

@tool
def report_hardware_issue(
    employee_id: str,
    issue: str,
) -> str:
    """
    Report hardware issue.
    """

    employees = _load_employees()

    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    tickets = _load_tickets()

    ticket_id = _generate_ticket_id(tickets)

    tickets.append(
         {
            "ticket_id": ticket_id,
            "employee_id": employee_id,
            "employee_name": employee["name"],
            "issue": issue,
            "status": "Open",
        }
    )

    _save_tickets(tickets)

    return (
        f"Hardware issue reported successfully.\n"
        f"Ticket ID : {ticket_id}"
    )

@tool
def check_system_status(system_name: str) -> str:
    """
    Check enterprise system status.
    """

    systems = _load_systems()

    system = next(
        (
            s
            for s in systems
            if s["system"].lower() == system_name.lower()
        ),
        None,
    )

    if system is None:
        return "System not found."

    return (
        f"{system['system']} status : "
        f"{system['status']}"
    )

IT_TOOLS = [
    reset_password,
    unlock_account,
    raise_it_ticket,
    check_ticket_status,
    request_software_installation,
    request_vpn_access,
    report_hardware_issue,
    check_system_status,
]

IT_READ_ONLY_TOOLS = [
    check_ticket_status,
    check_system_status,
]

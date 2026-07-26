from langchain_core.tools import tool 
from utils.file_handler import load_json,save_json 
from utils.paths import (EMPLOYEE_FILE,TICKET_FILE,SYSTEM_FILE)
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

def _find_employee(employee_id,employees):
    return next(
        (emp for emp in employees if emp["employee_id"]==employee_id),
        None,
    )


def _generate_ticket_id(tickets):
    """
    Generates ticket IDs like:
    IT001
    IT002
    IT003
    """

    if not tickets:
        return "IT001"
    numbers=[
        int(ticket["ticket_id"].replace("IT",""))
        for ticket in tickets
    ]
    next_number=max(numbers)+1 
    return f"IT{next_number:03d}"

@tool 
def reset_password(employee_id:str)->str:
    """
    Reset an employee's password.
    
    Args:
        employee_id:Employee ID 
    
    Returns:
        Success message.
    """

    employees=_load_employees()

    employee=_find_employee(employee_id,employees)

    if employee is None:
        return f"Employee not found."
    
    return (
        f"Password reset request has been successfully processed. "
        f"{employee['name']} ({employee_id})."
        f"A temporary password has been sent to {employee['email']}."
    )

@tool
def unlock_account(employee_id:str)->str:
    """
    Unlock a locked employee account.
    
    Args:
        employee_id:Employee ID 
        
    Returns:
        Success message.
    """

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
def raise_it_ticket(employee_id: str, issue: str) -> str:
    """
    Raise an IT support ticket.
    """

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

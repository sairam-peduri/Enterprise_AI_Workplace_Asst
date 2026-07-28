import re
from difflib import SequenceMatcher
from langchain_core.tools import tool
from src.utils.file_handler import load_json, save_json
from src.utils.paths import (
    EMPLOYEE_FILE,
    TRAVEL_FILE,
    TRAVEL_RATES_FILE,
    ITINERARY_FILE
)

"""
Travel Tools Module

This module contains travel support tools used by the Travel Agent.
These tools simulate common enterprise travel operations.
"""


def _load_employees():
    return load_json(EMPLOYEE_FILE)


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
            (emp for emp in _load_employees() if str(emp.get("employee_id", "")).upper() == emp_id),
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


def _load_travel_requests():
    return load_json(TRAVEL_FILE)


def _save_travel_requests(requests):
    save_json(TRAVEL_FILE, requests)


def _find_employee(employee_id, employees):
    employee_id = employee_id.strip().upper()
    return next(
        (emp for emp in employees if str(emp.get("employee_id", "")).upper() == employee_id),
        None,
    )


def _generate_request_id(requests):
    """
    Generates request IDs like:
    TR001
    TR002
    TR003
    """

    if not requests:
        return "TR001"

    numbers = []
    for request in requests:
        try:
            numbers.append(int(request.get("request_id", "").replace("TR", "")))
        except ValueError:
            continue

    next_number = max(numbers, default=0) + 1

    return f"TR{next_number:03d}"


@tool
def request_business_travel(
    employee_id: str,
    source: str,
    destination: str,
    start_date: str,
    end_date: str,
    purpose: str,
    confirmed: bool = False,
) -> str:
    """
    Submit a business travel request. Only submits when confirmed is True.
    """

    if not confirmed:
        return "Confirmation required. Show the travel request summary and ask the employee to reply yes before submitting."

    employees = _load_employees()

    employee = _find_employee(employee_id, employees)

    if employee is None:
        return "Employee not found."

    requests = _load_travel_requests()

    request_id = _generate_request_id(requests)

    requests.append(
        {
            "request_id": request_id,
            "employee_id": employee_id,
            "employee_name": employee["name"],
            "source": source,
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "purpose": purpose,
            "status": "Pending Approval",
        }
    )

    _save_travel_requests(requests)

    return (
        f"Travel request {request_id} has been submitted successfully "
        f"for {employee['name']}.\n"
        f"Status : Pending Approval"
    )


@tool
def estimate_budget(
    destination: str,
    days: int,
) -> str:
    """
    Estimate travel budget for a business trip.

    Args:
        destination: Destination city.
        days: Number of travel days.

    Returns:
        Estimated travel budget breakdown.
    """

    if days <= 0:
        return "Travel duration must be at least one day."
    rates = load_json(TRAVEL_RATES_FILE)

    city = next((value for key, value in rates.items() if key.lower() == destination.strip().lower()), None)

    if city is None:
        return (
            f"Travel rates for '{destination}' are not available.\n"
            f"Please choose a supported destination."
        )

    flight = city["flight"]*2
    hotel = city["hotel_per_day"] * days
    food = city["food_per_day"] * days
    transport = city["transport_per_day"] * days

    total = flight + hotel + food + transport

    return (
        f"Estimated Budget for {destination}\n\n"
        f"Flight : ₹{flight}\n"
        f"Hotel : ₹{hotel}\n"
        f"Food : ₹{food}\n"
        f"Local Transport : ₹{transport}\n"
        f"Total Estimated Cost : ₹{total}"
    )

@tool
def generate_travel_plan(
    destination: str,
    days: int,
    purpose: str = "Client Meeting",
) -> str:
    """
    Generate a business travel itinerary.
    """

    itineraries = load_json(ITINERARY_FILE)

    template = itineraries.get(purpose)

    if template is None:
        purpose = "Client Meeting"
        template = itineraries[purpose]

    plan = []

    plan.append("Business Travel Plan")
    plan.append(f"Destination : {destination}")
    plan.append(f"Purpose : {purpose}")
    plan.append(f"Duration : {days} day(s)")
    plan.append("")

    # Day 1
    plan.append("Day 1")
    for activity in template["arrival"]:
        plan.append(
            f"- {activity.replace('{destination}', destination)}"
        )

    # Middle Days
    if days > 2:
        for day in range(2, days):
            plan.append("")
            plan.append(f"Day {day}")

            for activity in template["business"]:
                plan.append(f"- {activity}")

    # Last Day
    if days >= 2:
        plan.append("")
        plan.append(f"Day {days}")

        for activity in template["departure"]:
            plan.append(
                f"- {activity.replace('{destination}', destination)}"
            )

    return "\n".join(plan)

@tool
def check_travel_status(request_id: str) -> str:
    """
    Check the status of a travel request.
    """

    requests = _load_travel_requests()

    request = next(
        (
            req
            for req in requests
            if req["request_id"] == request_id
        ),
        None,
    )

    if request is None:
        return "Travel request not found."

    return (
        f"Request ID : {request['request_id']}\n"
        f"Employee : {request['employee_name']}\n"
        f"Destination : {request['destination']}\n"
        f"Travel Dates : {request['start_date']} to {request['end_date']}\n"
        f"Purpose : {request['purpose']}\n"
        f"Status : {request['status']}"
    )


@tool
def cancel_travel_request(request_id: str, confirmed: bool = False) -> str:
    """
    Cancel a pending travel request. Only cancels when confirmed is True.
    """

    if not confirmed:
        return "Confirmation required. Show the cancellation summary and ask the employee to reply yes before cancelling."

    requests = _load_travel_requests()

    request = next(
        (
            req
            for req in requests
            if req["request_id"] == request_id
        ),
        None,
    )

    if request is None:
        return "Travel request not found."

    if request["status"] == "Cancelled":
        return "Travel request is already cancelled."

    request["status"] = "Cancelled"

    _save_travel_requests(requests)

    return (
        f"Travel request {request_id} has been cancelled successfully."
    )


TRAVEL_TOOLS = [
    request_business_travel,
    estimate_budget,
    generate_travel_plan,
    check_travel_status,
    cancel_travel_request,
]

TRAVEL_READ_ONLY_TOOLS = [
    estimate_budget,
    generate_travel_plan,
    check_travel_status,
]

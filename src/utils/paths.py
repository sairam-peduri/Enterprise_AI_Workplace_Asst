from pathlib import Path 

PROJECT_ROOT=Path(__file__).resolve().parents[2]

DATA_DIR=PROJECT_ROOT/"data"

IT_DATA_DIR=DATA_DIR/"it"

HR_DATA_DIR=DATA_DIR/"hr"

FINANCE_DATA_DIR=DATA_DIR/"finance"

TRAVEL_DATA_DIR=DATA_DIR/"travel"
TRAVEL_FILE = TRAVEL_DATA_DIR / "travel_requests.json"
TRAVEL_RATES_FILE = TRAVEL_DATA_DIR / "travel.json"
ITINERARY_FILE = TRAVEL_DATA_DIR /"itenararies.json"

EMPLOYEE_FILE=IT_DATA_DIR/"employees.json"

SYSTEM_FILE=IT_DATA_DIR/"systems.json"

TICKET_FILE=IT_DATA_DIR/"tickets.json"

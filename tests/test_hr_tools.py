import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tools.hr_tools import apply_leave, check_leave_balance, holiday_calendar

print(
    apply_leave.invoke(
        {
            "employee_id": "EMP001",
            "days": 2,
            "leave_type": "Annual",
            "reason": "Family visit",
        }
    )
)

print(
    check_leave_balance.invoke(
        {"employee_id": "EMP001"}
    )
)

print(
    holiday_calendar.invoke({})
)
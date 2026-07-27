from src.tools.travel_tools import (
    estimate_budget,
    generate_travel_plan,
    request_business_travel,
    check_travel_status,
)

print("===== Budget =====")
print(estimate_budget.invoke({
    "destination": "Bangalore",
    "days": 3
}))

print("\n===== Travel Plan =====")
print(generate_travel_plan.invoke({
    "destination": "Mumbai",
    "days": 2
}))

print("\n===== Travel Request =====")
print(request_business_travel.invoke({
    "employee_id": "EMP001",
    "source": "Hyderabad",
    "destination": "Delhi",
    "start_date": "2026-08-10",
    "end_date": "2026-08-13",
    "purpose": "Client Meeting"
}))

print("\n===== Status =====")
print(check_travel_status.invoke({
    "request_id": "TR001"
}))
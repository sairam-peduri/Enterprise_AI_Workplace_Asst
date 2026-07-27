from src.tools.it_tools import reset_password

print(
    reset_password.invoke(
        {"employee_id": "EMP001"}
    )
)

print(
    reset_password.invoke(
        {"employee_id": "EMP999"}
    )
)
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]/"src"))

from tools.it_tools import reset_password

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
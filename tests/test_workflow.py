import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage
from src.graph.workflow import workflow

state = {
    "messages": [
        HumanMessage(content="Reset password for employee EMP001")
    ]
}

response = workflow.invoke(state)

print("\n===== FINAL RESPONSE =====\n")

for message in response["messages"]:
    print(message)
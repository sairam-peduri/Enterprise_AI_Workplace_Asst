from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class EnterpriseState(TypedDict):
    """
    Shared state passed between all agents in the workflow.
    """

    messages: Annotated[list, add_messages]
    employee_id: str | None
    employee_name: str | None
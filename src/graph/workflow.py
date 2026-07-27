"""LangGraph workflow that routes a request to one specialist agent."""

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, ToolMessage

from src.agents.finance_agent import finance_agent
from src.agents.hr_agent import hr_agent
from src.agents.it_agent import it_agent
from src.agents.knowledge_agent import knowledge_agent_node
from src.agents.travel_agent import travel_agent
from src.graph.supervisor import supervisor_node
from src.graph.tool_dispatch import dispatch_known_request
from src.state.state import EnterpriseState
from src.utils.logging_config import record_event


def _record_tool_messages(messages: list) -> None:
    """Log tools the model actually invoked in an agent execution."""
    for message in messages:
        if isinstance(message, ToolMessage):
            record_event("tool_completed", tool=message.name or "unnamed_tool")


def _invoke_agent(agent, state: EnterpriseState) -> dict:
    """Invoke a specialist while returning only messages added by that agent."""
    # The model handles conversational requests; deterministic dispatch protects
    # clear actions from invented answers when a local model skips a tool call.
    route = {
        it_agent: "it_agent",
        hr_agent: "hr_agent",
        finance_agent: "finance_agent",
        travel_agent: "travel_agent",
    }.get(agent)
    if route:
        latest = str(getattr(state["messages"][-1], "content", ""))
        tool_result = dispatch_known_request(route, latest)
        if tool_result:
            record_event("agent_completed", agent=route, execution="deterministic_tool")
            return {"messages": [tool_result]}
    agent_name = route or "unknown_agent"
    record_event("agent_started", agent=agent_name, execution="llm")
    result = agent.invoke({"messages": state["messages"]})
    existing_count = len(state["messages"])
    new_messages = result["messages"][existing_count:]
    added_messages = new_messages or [result["messages"][-1]]
    _record_tool_messages(added_messages)
    record_event("agent_completed", agent=agent_name, execution="llm")
    return {"messages": added_messages}


def it_node(state: EnterpriseState) -> dict:
    return _invoke_agent(it_agent, state)


def hr_node(state: EnterpriseState) -> dict:
    return _invoke_agent(hr_agent, state)


def finance_node(state: EnterpriseState) -> dict:
    return _invoke_agent(finance_agent, state)


def travel_node(state: EnterpriseState) -> dict:
    return _invoke_agent(travel_agent, state)


def knowledge_node(state: EnterpriseState) -> dict:
    record_event("agent_started", agent="knowledge_agent", execution="llm")
    result = knowledge_agent_node(state)
    _record_tool_messages(result["messages"])
    record_event("agent_completed", agent="knowledge_agent", execution="llm")
    return result


def general_node(state: EnterpriseState) -> dict:
    record_event("agent_started", agent="general_agent", execution="built_in")
    record_event("agent_completed", agent="general_agent", execution="built_in")
    return {
        "messages": [
            AIMessage(
                content=(
                    "Hello! I’m Enterprise AI. Tell me what you need, and my supervisor "
                    "will route your request to the right workplace specialist."
                )
            )
        ]
    }


builder = StateGraph(EnterpriseState)
builder.add_node("general_agent", general_node)
builder.add_node("it_agent", it_node)
builder.add_node("hr_agent", hr_node)
builder.add_node("finance_agent", finance_node)
builder.add_node("travel_agent", travel_node)
builder.add_node("knowledge_agent", knowledge_node)

builder.add_conditional_edges(
    START,
    supervisor_node,
    {
        "general_agent": "general_agent",
        "it_agent": "it_agent",
        "hr_agent": "hr_agent",
        "finance_agent": "finance_agent",
        "travel_agent": "travel_agent",
        "knowledge_agent": "knowledge_agent",
    },
)

for node_name in (
    "general_agent",
    "it_agent",
    "hr_agent",
    "finance_agent",
    "travel_agent",
    "knowledge_agent",
):
    builder.add_edge(node_name, END)

workflow = builder.compile()

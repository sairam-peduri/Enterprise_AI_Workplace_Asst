from langgraph.graph import StateGraph, START, END

from src.state.state import EnterpriseState
from src.graph.supervisor import supervisor_node
from src.agents.it_agent import it_agent

def it_node(state: EnterpriseState):
    response = it_agent.invoke(state)

    return {
        "messages": response["messages"]
    }

def hr_node(state: EnterpriseState):
    """
    Placeholder until HR Agent is integrated.
    """
    return state


def finance_node(state: EnterpriseState):
    """
    Placeholder until Finance Agent is integrated.
    """
    return state


def travel_node(state: EnterpriseState):
    """
    Placeholder until Travel Agent is integrated.
    """
    return state


def knowledge_node(state: EnterpriseState):
    """
    Placeholder until Knowledge Agent is integrated.
    """
    return state


builder = StateGraph(EnterpriseState)

builder.add_node("it_agent", it_node)
builder.add_node("hr_agent", hr_node)
builder.add_node("finance_agent", finance_node)
builder.add_node("travel_agent", travel_node)
builder.add_node("knowledge_agent", knowledge_node)

builder.add_conditional_edges(
    START,
    supervisor_node,
    {
    "it_agent": "it_agent",
    "hr_agent": "hr_agent",
    "finance_agent": "finance_agent",
    "travel_agent": "travel_agent",
    "knowledge_agent": "knowledge_agent",
    },
)

builder.add_edge("it_agent", END)
builder.add_edge("hr_agent", END)
builder.add_edge("finance_agent", END)
builder.add_edge("travel_agent", END)
builder.add_edge("knowledge_agent", END)

workflow = builder.compile()
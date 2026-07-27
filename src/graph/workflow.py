from langgraph.graph import StateGraph, START, END

from src.agents import hr_agent
from src.state.state import EnterpriseState
from src.graph.supervisor import supervisor_node
from src.agents.it_agent import it_agent
<<<<<<< HEAD
from src.agents.hr_agent import hr_agent    
=======
from src.agents.finance_agent import finance_agent
>>>>>>> 73aac7e1e53ecc6f5b0423cbb0541a9affbb4ebe

def it_node(state: EnterpriseState):
    response = it_agent.invoke(state)

    return {
        "messages": response["messages"]
    }

def hr_node(state: EnterpriseState):
    response = hr_agent.invoke(state)
    
    return {
            "messages": response["messages"]
        }


def finance_node(state: EnterpriseState):
    response = finance_agent.invoke(state)

    return {
        "messages": [response["messages"][-1]]
    }

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
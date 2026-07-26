from langgraph.graph import StateGraph, START, END

from state.state import EnterpriseState
from graph.supervisor import supervisor_node
from agents.it_agent import it_agent

def it_node(state: EnterpriseState):
    response = it_agent.invoke(state)

    return {
        "messages": response["messages"]
    }

builder = StateGraph(EnterpriseState)

builder.add_node("it_agent", it_node)

builder.add_conditional_edges(
    START,
    supervisor_node,
    {
        "it_agent": "it_agent",
    },
)

builder.add_edge("it_agent", END)

workflow = builder.compile()
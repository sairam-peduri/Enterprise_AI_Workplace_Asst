"""RAG-backed knowledge specialist."""

from functools import lru_cache

from langchain.agents import create_agent

from src.tools.knowledge_tools import search_policy
from src.utils.models import llm
KNOWLEDGE_AGENT_SYSTEM_PROMPT = """
You are the EnterpriseCorp Knowledge Base Assistant. Answer only from company
documents returned by the search_policy tool. Always search before answering.
Include the source file and page number supplied by the tool. If the search has
no relevant result, say: "I cannot find this information in the official company documents."
Do not invent policies or citations.
"""


@lru_cache(maxsize=1)
def get_knowledge_agent():
    return create_agent(
        model=llm,
        tools=[search_policy],
        system_prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT,
    )


def knowledge_agent_node(state: dict) -> dict:
    result = get_knowledge_agent().invoke({"messages": state["messages"]})
    existing_count = len(state["messages"])
    new_messages = result["messages"][existing_count:]
    return {"messages": new_messages or [result["messages"][-1]]}

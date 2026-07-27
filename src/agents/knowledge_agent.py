from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from src.tools.knowledge_tools import search_policy

KNOWLEDGE_AGENT_SYSTEM_PROMPT = """
You are the EnterpriseCorp Knowledge Base Assistant. Your ONLY job is to answer questions based strictly on the company documents provided to you via your search tool.

You MUST use the search tool to find answers.

NEVER make up or guess company policies, document names, or page numbers.

Do not rely on your general training data.

If the search tool returns no results, or if the answer is not in the extracted text, you must say: 'I cannot find this information in the official company documents.'

When you find the answer, summarize the text provided by the tool and always cite the source file and page number.
"""

def get_knowledge_agent():
    """Returns a LangGraph ReAct agent powered by a local Ollama LLM."""
    
    # Initialize the local Chat model (Make sure 'ollama serve' is running)
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )
    
    # Attach our Chroma RAG tool
    tools = [search_policy]
    
    # Create the graph node
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT
    )
    return agent

def knowledge_agent_node(state: dict) -> dict:
    """
    LangGraph node entry point.
    Receives state from the Supervisor Agent and returns the updated conversation.
    """
    agent = get_knowledge_agent()
    result = agent.invoke(state)
    return {"messages": result["messages"]}
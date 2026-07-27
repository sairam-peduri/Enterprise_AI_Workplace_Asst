from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.prompts.travel_prompt import TRAVEL_AGENT_PROMPT
from src.tools.travel_tools import TRAVEL_TOOLS

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

travel_agent = create_agent(
    model=llm,
    tools=TRAVEL_TOOLS,
    system_prompt=TRAVEL_AGENT_PROMPT,
)
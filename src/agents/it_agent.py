from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.prompts.it_prompt import IT_AGENT_PROMPT
from src.tools.it_tools import IT_TOOLS

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

it_agent = create_agent(
    model=llm,
    tools=IT_TOOLS,
    system_prompt=IT_AGENT_PROMPT,
)
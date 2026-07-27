from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.prompts.finance_prompt import FINANCE_AGENT_PROMPT
from src.tools.finance_tools import FINANCE_TOOLS


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)


finance_agent = create_agent(
    model=llm,
    tools=FINANCE_TOOLS,
    system_prompt=FINANCE_AGENT_PROMPT,
)
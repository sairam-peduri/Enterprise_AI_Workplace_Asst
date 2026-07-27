# HR Agent
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.prompts.hr_prompt import HR_AGENT_PROMPT
from src.tools.hr_tools import HR_TOOLS

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

hr_agent = create_agent(
    model=llm,
    tools=HR_TOOLS,
    system_prompt=HR_AGENT_PROMPT,
)
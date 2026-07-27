from langchain.agents import create_agent

from src.prompts.it_prompt import IT_AGENT_PROMPT
from src.tools.it_tools import IT_TOOLS
from src.utils.models import llm

it_agent = create_agent(
    model=llm,
    tools=IT_TOOLS,
    system_prompt=IT_AGENT_PROMPT,
)

from langchain.agents import create_agent

from src.prompts.finance_prompt import FINANCE_AGENT_PROMPT
from src.tools.finance_tools import FINANCE_TOOLS
from src.utils.models import llm


finance_agent = create_agent(
    model=llm,
    tools=FINANCE_TOOLS,
    system_prompt=FINANCE_AGENT_PROMPT,
)

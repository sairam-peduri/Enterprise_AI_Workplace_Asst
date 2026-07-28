from langchain.agents import create_agent

from src.prompts.travel_prompt import TRAVEL_AGENT_PROMPT
from src.tools.travel_tools import TRAVEL_READ_ONLY_TOOLS
from src.utils.models import llm

travel_agent = create_agent(
    model=llm,
    tools=TRAVEL_READ_ONLY_TOOLS,
    system_prompt=TRAVEL_AGENT_PROMPT,
)

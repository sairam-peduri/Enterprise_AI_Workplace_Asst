from src.agents.it_agent import it_agent

response = it_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Reset password for employee EMP001"
            }
        ]
    }
)

print(response)
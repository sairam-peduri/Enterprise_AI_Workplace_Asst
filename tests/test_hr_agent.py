from src.agents.hr_agent import hr_agent

response = hr_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Apply leave for EMP001 for 2 days"
            }
        ]
    }
)

print(response)

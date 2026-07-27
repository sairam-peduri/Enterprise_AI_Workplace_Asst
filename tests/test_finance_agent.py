from src.agents.finance_agent import finance_agent


def test_finance_agent():
    response = finance_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I am employee EMP001. "
                        "What is the reimbursement status of EXP1002?"
                    ),
                }
            ]
        }
    )

    print("\n--- FINANCE AGENT RESPONSE ---\n")

    print(response["messages"][-1].content)


if __name__ == "__main__":
    test_finance_agent()
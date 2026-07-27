from src.agents.finance_agent import finance_agent


def print_last_message(response):
    print("\nFINANCE AGENT:")
    print(response["messages"][-1].content)


# ---------------------------------------
# STEP 1
# Employee asks to submit an expense
# ---------------------------------------

messages = [
    {
        "role": "user",
        "content": (
            "I want to submit an expense of ₹1200 "
            "for a cab to a client meeting."
        ),
    }
]

response = finance_agent.invoke({"messages": messages})

print("\n--- STEP 1: INITIAL REQUEST ---")
print_last_message(response)


# Preserve conversation history
messages = response["messages"]


# ---------------------------------------
# STEP 2
# Employee provides missing information
# ---------------------------------------

messages.append(
    {
        "role": "user",
        "content": (
            "My user ID is EMP001. "
            "The expense date was 2026-07-26 "
            "and I have the receipt."
        ),
    }
)

response = finance_agent.invoke({"messages": messages})

print("\n--- STEP 2: PROVIDE DETAILS ---")
print_last_message(response)


# Preserve conversation again
messages = response["messages"]


# ---------------------------------------
# IMPORTANT:
# We intentionally STOP here.
#
# We are NOT saying Yes yet.
# Therefore submit_expense should NOT run.
# ---------------------------------------


if __name__ == "__main__":
    print(
        "\n--- TEST COMPLETE ---\n"
        "Check expenses.json and verify that no new "
        "expense was created yet."
    )
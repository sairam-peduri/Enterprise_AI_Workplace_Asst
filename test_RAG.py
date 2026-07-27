"""Manual command-line RAG demo (not part of automated test collection)."""

from src.agents.knowledge_agent import get_knowledge_agent


def main():
    agent = get_knowledge_agent()
    print("EnterpriseCorp Knowledge Base is ready. Type 'exit' to close.")
    while True:
        question = input("\nEmployee: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Closing the Knowledge Base. Goodbye!")
            break
        response = agent.invoke({"messages": [("user", question)]})
        print(response["messages"][-1].content)


if __name__ == "__main__":
    main()

from langchain_core.messages import HumanMessage

from src.graph.workflow import workflow


def test_finance_workflow():

    state = {
        "messages": [
            HumanMessage(
                content=(
                    "For user ID EMP001, "
                    "what is the reimbursement status of EXP1002?"
    )
)
        ]
    }

    result = workflow.invoke(state)

    print("\n--- FINANCE WORKFLOW RESPONSE ---\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    test_finance_workflow()
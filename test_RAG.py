# # test_rag.py (Run this in your terminal)
# from src.agents.knowledge_agent import knowledge_agent_node

# test_state = {
#     "messages": [("user", "Summarize the remote work policy.")]
# }

# response = knowledge_agent_node(test_state)
# print(response["messages"][-1].content)

from src.agents.knowledge_agent import get_knowledge_agent 

# Initialize your agent
agent = get_knowledge_agent()

print("\n🤖 EnterpriseCorp Knowledge Base is ready!")
print("Type 'exit' or 'quit' to close the chat.")
print("-" * 50)

# Start the interactive chat loop
while True:
    user_question = input("\nEmployee: ")
    
    if user_question.lower() in ['exit', 'quit', 'q']:
        print("Closing the Knowledge Base. Goodbye!")
        break
        
    print("\n--- AGENT THOUGHT PROCESS ---")
    
    # Use stream() instead of invoke() to see the intermediate steps
    for chunk in agent.stream({"messages": [("user", user_question)]}):
        # Print each step the agent takes (Tool calls, tool results, final answer)
        for node_name, node_data in chunk.items():
            print(f"[{node_name}] is working...\n")
            
            # Print the content of the latest message in this node (if it has text)
            if "messages" in node_data and len(node_data["messages"]) > 0:
                content = node_data["messages"][-1].content
                if content:
                    print(content)
                
    print("\n-----------------------------")
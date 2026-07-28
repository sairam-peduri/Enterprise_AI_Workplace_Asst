from src.agents.travel_agent import travel_agent

response = travel_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                #"content": "Generate a business travel plan for Chennai for 3 days for client meeting."
                "content": "Estimate travel budget for Hyderabad for 5 days."
                #"content":"Check travel request status TR003."
                #"content":"Check travel request status TR999."
                # "content":"Cancel travel request TR003."
                
            }
        ]
    }
)

# # print(response)
# print(response["messages"][-1].content)

from langchain_core.messages import ToolMessage

for message in response["messages"]:
    if isinstance(message, ToolMessage):
        print(message.content)



# from langchain_core.messages import ToolMessage
# from src.agents.travel_agent import travel_agent

# response = travel_agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": """Submit a business travel request.

# Employee ID: EMP002
# Source: Hyderabad
# Destination: Bangalore
# Start Date: 2026-08-15
# End Date: 2026-08-18
# Purpose: Client Meeting
# """
#             }
#         ]
#     }
# )

# for message in response["messages"]:
#     if isinstance(message, ToolMessage):
#         print(message.content)
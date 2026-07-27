import streamlit as st
import traceback
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from src.graph.workflow import workflow
from src.ui.components import agent_badge


# ------------------------------------------------------
# Initialize Chat
# ------------------------------------------------------

def initialize_chat():

    if "messages" not in st.session_state:
        st.session_state.messages = []


# ------------------------------------------------------
# Display Previous Messages
# ------------------------------------------------------

def display_chat():

    for message in st.session_state.messages:

        if isinstance(message, HumanMessage):

            with st.chat_message("user", avatar="👤"):

                st.markdown(message.content)

        elif isinstance(message, AIMessage):

            with st.chat_message("assistant", avatar="🤖"):

                agent_badge("IT")

                st.markdown(message.content)


# ------------------------------------------------------
# Handle User Input
# ------------------------------------------------------

def handle_user_input():

    prompt = st.chat_input(
        "Ask Enterprise AI anything..."
    )

    if not prompt:
        return

    user_message = HumanMessage(content=prompt)

    st.session_state.messages.append(user_message)

    with st.chat_message("user", avatar="👤"):

        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("🤖 Enterprise AI is processing your request..."):

            try:

                response = workflow.invoke(
                    {
                        "messages": st.session_state.messages
                    }
                )

                ai_messages = [

                    msg

                    for msg in response["messages"]

                    if isinstance(msg, AIMessage)

                ]

                if ai_messages:

                    final_response = ai_messages[-1]

                    agent_badge("IT")

                    st.markdown(final_response.content)

                    st.session_state.messages.append(
                        final_response
                    )

                else:

                    st.warning(
                        "No response generated."
                    )

            except Exception as e:

                # st.error(
                #     f" {str(e)}"
                # )
                st.code(traceback.format_exc())
import streamlit as st

from src.ui.styles import load_css
from src.ui.sidebar import render_sidebar
from src.ui.homepage import render_homepage
from src.ui.chat import (
    initialize_chat,
    display_chat,
    handle_user_input,
)
from src.ui.components import footer

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise AI Workplace Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------

load_css()

# --------------------------------------------------
# Initialize Chat
# --------------------------------------------------

initialize_chat()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

render_sidebar()

# --------------------------------------------------
# Homepage
# --------------------------------------------------

if len(st.session_state.messages) == 0:

    render_homepage()

else:

    st.markdown(
        """
        <div style="text-align:center;padding-bottom:20px;">

        <h1 style="color:#2563EB;">

        🤖 Enterprise AI

        </h1>

        <p style="color:gray;">

        AI-powered workplace assistant

        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# Chat
# --------------------------------------------------

display_chat()

handle_user_input()

# --------------------------------------------------
# Footer
# --------------------------------------------------

footer()
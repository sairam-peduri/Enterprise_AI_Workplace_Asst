from datetime import datetime
import streamlit as st

from src.ui.components import (
    section_title,
    suggestion,
)


# --------------------------------------------------------
# Greeting
# --------------------------------------------------------

def get_greeting():

    hour = datetime.now().hour

    if hour < 12:
        return "🌅 Good Morning"

    elif hour < 17:
        return "☀️ Good Afternoon"

    else:
        return "🌙 Good Evening"


# --------------------------------------------------------
# Homepage
# --------------------------------------------------------

def render_homepage():

    greeting = get_greeting()

    st.markdown(
        f"""
        <div style="text-align:center; padding:10px 0 30px 0;">

            <h1 style="
                color:#2563EB;
                font-size:48px;
                margin-bottom:0px;
            ">
                🤖 Enterprise AI
            </h1>

            <h3 style="
                color:#475569;
                font-weight:400;
                margin-top:8px;
            ">
                Workplace Assistant
            </h3>

            <p style="
                color:#64748B;
                font-size:22px;
                margin-top:20px;
            ">
                {greeting}
            </p>

            <p style="
                color:#94A3B8;
                font-size:18px;
            ">
                How can I help you today?
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------
    # Stats
    # ----------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Available Modules",
            "1",
            "+4 Soon"
        )

    with c2:
        st.metric(
            "AI Model",
            "Llama 3.2"
        )

    with c3:
        st.metric(
            "Framework",
            "LangGraph"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # Quick Suggestions
    # ----------------------------------------------------

    section_title("💡 Try Asking")

    col1, col2 = st.columns(2)

    with col1:

        suggestion("Reset password for EMP001")

        suggestion("Unlock account EMP002")

        suggestion("Raise hardware issue for EMP003")

    with col2:

        suggestion("Check ticket TKT1001")

        suggestion("Request VPN access")

        suggestion("Install Microsoft Office")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # Features
    # ----------------------------------------------------

    section_title("🚀 Enterprise Modules")

    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:
        st.success("💻 IT")

    with f2:
        st.info("👨‍💼 HR")

    with f3:
        st.info("💰 Finance")

    with f4:
        st.info("✈️ Travel")

    with f5:
        st.info("📚 Knowledge")
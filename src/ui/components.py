import streamlit as st


# ---------------------------------------------------
# Section Title
# ---------------------------------------------------

def section_title(title: str):
    st.markdown(
        f"""
        <h3 style="
        margin-top:25px;
        color:#1E3A8A;
        ">
        {title}
        </h3>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# Agent Badge
# ---------------------------------------------------

def agent_badge(agent_name: str):

    icons = {
        "IT": "💻",
        "HR": "👨‍💼",
        "Finance": "💰",
        "Travel": "✈️",
        "Knowledge": "📚",
    }

    icon = icons.get(agent_name, "🤖")

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 14px;
            border-radius:30px;
            background:#DBEAFE;
            color:#1D4ED8;
            font-weight:600;
            margin-bottom:10px;
        ">
        {icon} {agent_name} Agent
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# Status Badge
# ---------------------------------------------------

def status_badge(status="Connected"):

    color = "#16A34A" if status == "Connected" else "#DC2626"

    icon = "🟢" if status == "Connected" else "🔴"

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 12px;
            border-radius:20px;
            background:#F8FAFC;
            border:1px solid #CBD5E1;
            font-size:14px;
        ">
        {icon}
        <span style="color:{color};font-weight:600;">
        Ollama {status}
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# Suggestion Card
# ---------------------------------------------------

def suggestion(text):

    st.markdown(
        f"""
        <div style="
            border:1px solid #E5E7EB;
            padding:14px;
            border-radius:12px;
            margin-bottom:12px;
            background:white;
            transition:0.3s;
        ">
        💬 {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# Footer
# ---------------------------------------------------

def footer():

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        """
        <center>

        Enterprise AI Workplace Assistant

        <br>

        <small>

        Built with Streamlit • LangGraph • LangChain • Ollama

        </small>

        </center>
        """,
        unsafe_allow_html=True,
    )
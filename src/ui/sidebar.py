import streamlit as st


def render_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <div style="text-align:center;padding-top:10px;">

            <h2 style="margin-bottom:0;">
                🤖 Enterprise AI
            </h2>

            <p style="color:gray;margin-top:0;">
                Workplace Assistant
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("### 📂 Modules")

        modules = [
            ("💻", "IT Support", True),
            ("👨‍💼", "HR", False),
            ("💰", "Finance", False),
            ("✈️", "Travel", False),
            ("📚", "Knowledge", False),
        ]

        for icon, module, active in modules:

            if active:

                st.success(f"{icon} {module}")

            else:

                st.caption(f"{icon} {module}")

        st.divider()

        if st.button(
            "🗑 New Chat",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()

        st.divider()

        st.markdown("### ⚙️ System")

        st.success("🟢 Ollama Connected")

        st.info("🕸 LangGraph Ready")

        st.info("🦜 LangChain Ready")

        st.divider()

        st.markdown(
            """
            <center>

            <small>

            Enterprise AI Workplace Assistant

            <br>

            Version 2.0

            </small>

            </center>
            """,
            unsafe_allow_html=True,
        )
import streamlit as st


def load_css():
    st.markdown("""
    <style>

    /* -----------------------------
       Hide Streamlit Branding
    ------------------------------*/

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}

    /* -----------------------------
       Main Page
    ------------------------------*/

    .block-container{
        padding-top:1.5rem;
        padding-bottom:1rem;
        padding-left:2rem;
        padding-right:2rem;
    }

    /* -----------------------------
       Sidebar
    ------------------------------*/

    section[data-testid="stSidebar"]{
        background:#f8fafc;
        border-right:1px solid #E5E7EB;
    }

    /* -----------------------------
       Title
    ------------------------------*/

    .title{

        font-size:42px;

        font-weight:700;

        color:#2563EB;

        margin-bottom:0px;

    }

    .subtitle{

        color:#64748B;

        font-size:17px;

        margin-bottom:30px;

    }

    /* -----------------------------
       Welcome Box
    ------------------------------*/

    .welcome{

        background:white;

        border-radius:15px;

        padding:22px;

        border:1px solid #E5E7EB;

        margin-bottom:25px;

        box-shadow:0px 3px 12px rgba(0,0,0,.05);

    }

    .welcome h3{

        margin-top:0;

    }

    /* -----------------------------
       Footer
    ------------------------------*/

    .footer{

        margin-top:40px;

        text-align:center;

        color:#94A3B8;

        font-size:14px;

    }

    /* -----------------------------
       Chat Bubbles
    ------------------------------*/

    [data-testid="stChatMessage"]{

        border-radius:15px;

        padding:8px;

    }

    /* -----------------------------
       Buttons
    ------------------------------*/

    .stButton>button{

        width:100%;

        border-radius:10px;

        height:45px;

        font-weight:600;

    }

    </style>
    """, unsafe_allow_html=True)
# auth.py
import streamlit as st

def require_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.error("🔐 Please log in first.")
        st.stop()  # Prevents page from continuing

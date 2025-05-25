import streamlit as st
from daily1 import show_daily
from range import show_range
from daily import show_about

def app_content():
    st.title("📊 Daily Energy Trading Dashboard")

    # Password check
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        password = st.text_input("Enter password:", type="password")
        if password == st.secrets["general"]["password"]:
            st.session_state.authenticated = True
            st.experimental_rerun()
        elif password:
            st.error("Incorrect password")
        return

    # Navigation
    page = st.sidebar.selectbox("Navigate", ["📊 Daily Dashboard", "📈 Range Dashboard", "ℹ️ About"])

    
    if page == "📊 Daily Dashboard":
        show_daily()
    elif page == "📈 Range Dashboard":
        show_range()
    elif page == "ℹ️ About":
        show_about()

if __name__ == "__main__":
    app_content()

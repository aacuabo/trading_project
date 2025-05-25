import streamlit as st
from dashboard import show_dashboard
from about import show_about
from upload import show_upload

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
    page = st.sidebar.selectbox("Navigate", ["📊 Dashboard", "Upload", "ℹ️ About"])

    
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "Upload":
        show_upload()
    elif page == "ℹ️ About":
        show_about()

if __name__ == "__main__":
    app_content()

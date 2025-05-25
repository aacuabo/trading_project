# main.py
import streamlit as st

def authenticate():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if (
            username == st.secrets["auth"]["username"]
            and password == st.secrets["auth"]["password"]
        ):
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        st.switch_page("pages/📊 Daily Dashboard.py")  # Or simplified name
    else:
        authenticate()

if __name__ == "__main__":
    main()

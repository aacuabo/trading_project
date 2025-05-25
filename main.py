import streamlit as st

# Login authentication from secrets
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
            st.rerun()
        else:
            st.error("Invalid credentials")

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        authenticate()
    else:
        # Navigate to the main dashboard
        st.switch_page("📊 Daily Dashboard.py")

if __name__ == "__main__":
    main()

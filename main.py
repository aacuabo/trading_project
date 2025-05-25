# main_app.py
import streamlit as st
import src.dashboard as dashboard
import src.reports as reports # Import your other secure page modules

# --- Authentication Logic ---
def authenticate_user():
    st.title("🔐 Login")
    st.info("Please enter your credentials to access the application.")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
        if (
            username == st.secrets["auth"]["username"]
            and password == st.secrets["auth"]["password"]
        ):
            st.session_state.authenticated = True
            st.session_state.current_page = "Dashboard" # Set default page after login
            st.success("Login successful!")
            st.rerun() # Re-run the app to update UI
        else:
            st.error("Invalid credentials. Please try again.")

# --- Main App Logic ---
def main():
    # Initialize session state for authentication and current page
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Login" # Default to login

    if st.session_state.authenticated:
        # User is authenticated: Show sidebar navigation and selected page content
        st.sidebar.title("Secure Navigation")

        # Define your secure pages as a dictionary for easy iteration
        secure_pages = {
            "Dashboard": dashboard.show_dashboard,
            "Reports": reports.show_reports,
            # Add more pages here: "Page Name": module.function_name
        }

        # Create sidebar buttons for navigation
        for page_name in secure_pages.keys():
            if st.sidebar.button(page_name, key=f"nav_{page_name.lower().replace(' ', '_')}"):
                st.session_state.current_page = page_name

        st.sidebar.markdown("---") # Separator
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.authenticated = False
            st.session_state.current_page = "Login" # Reset current page on logout
            st.success("You have been logged out.")
            st.rerun() # Re-run to show login page

        # Display the content of the currently selected secure page
        if st.session_state.current_page in secure_pages:
            secure_pages[st.session_state.current_page]()
        else:
            st.error("Page not found or invalid selection.") # Fallback for unexpected state

    else:
        # User is not authenticated: Only show the login screen
        authenticate_user()

if __name__ == "__main__":
    main()

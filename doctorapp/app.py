import streamlit as st
from pages import admin, dashboard, login

# def main():
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False

#     if st.session_state.logged_in:
#         dashboard.run()
#     else:
#         login.run()


# if __name__ == "__main__":
#     main()


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "role" not in st.session_state:
        st.session_state.role = None

    st.sidebar.title("Agentic Doctor")
    choice = st.sidebar.radio("Navigate", ["Login", "Dashboard", "Admin"], index=0)

    if choice == "Login":
        login.run()
    elif choice == "Dashboard":
        if st.session_state.logged_in:
            dashboard.run()
        else:
            st.warning("Please login first.")
    elif choice == "Admin":
        if st.session_state.get("role") == "admin":
            admin.run()
        else:
            st.warning("Admins only.")


if __name__ == "__main__":
    main()

import streamlit as st
from pages import login, dashboard


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        dashboard.run()
    else:
        login.run()


if __name__ == "__main__":
    main()

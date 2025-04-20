import streamlit as st


def run():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "doctor" and password == "ai123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")

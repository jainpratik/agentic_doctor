import json

import bcrypt
import streamlit as st

# def run():
#     st.title("🔐 Login")

#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")

#     if st.button("Login"):
#         if username == "doctor" and password == "ai123":
#             st.session_state.logged_in = True
#         else:
#             st.error("Invalid username or password")


def verify_user(username, password):
    with open("data/users.json", "r") as f:
        users = json.load(f)
    for user in users:
        if user["username"] == username and bcrypt.checkpw(
            password.encode(), user["password"].encode()
        ):
            return True, user["role"]
    return False, None


def run():
    st.title("🔐 Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        valid, role = verify_user(username, password)
        if valid:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.success(f"Logged in as {role}")
        else:
            st.error("Invalid credentials")

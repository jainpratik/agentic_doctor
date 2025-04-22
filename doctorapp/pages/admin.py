import json

import bcrypt
import streamlit as st


def run():
    st.title("👥 Admin User Panel")
    st.write("Manage users and roles for the Agentic Doctor application.")
    st.subheader("Add New User")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["doctor", "nurse", "admin"])

    if st.button("Add User"):
        if new_user and new_pass:
            with open("data/users.json", "r") as f:
                users = json.load(f)
            hashed_pw = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
            users.append({"username": new_user, "password": hashed_pw, "role": role})
            with open("data/users.json", "w") as f:
                json.dump(users, f, indent=4)
            st.success("User added successfully.")
        else:
            st.error("Enter valid credentials.")

    st.subheader("Existing Users")
    with open("data/users.json", "r") as f:
        users = json.load(f)
        for u in users:
            st.markdown(f"- **{u['username']}** ({u['role']})")

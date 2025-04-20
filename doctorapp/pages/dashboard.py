import json

import streamlit as st
from utils.backend import (
    agent_response,
    color_code_cells,
    generate_care_plan,
    get_patient_events,
    get_patient_vitals,
    parse_lab_pdf,
)

# from PyPDF2 import PdfReader


def run():

    st.set_page_config(page_title="Agentic Doctor", layout="wide")
    st.title("🩺 Agentic Doctor Dashboard")
    st.markdown("An AI-powered assistant for clinical reasoning and decision support.")

    # Sidebar: Patient Selection
    with st.sidebar:
        st.markdown("## 🧑‍⚕️ Select Patient")
        st.markdown(
            "Select a patient from the EMR database to view their details and lab reports."
        )

    # Load EMR patients
    with open("data/patients.json", "r") as f:
        emr_patients = json.load(f)

        patient_map = {f"{p['id']} - {p['name']}": p for p in emr_patients}
        selected_id = st.sidebar.selectbox(
            "📋 Select Patient", list(patient_map.keys())
        )
        patient = patient_map[selected_id]

        st.sidebar.text_input("Patient ID", value=patient["id"], disabled=True)
        st.sidebar.text_input("Name", value=patient["name"])
        st.sidebar.number_input("Age", value=patient["age"])
        st.sidebar.selectbox(
            "Sex", ["Male", "Female"], index=0 if patient["sex"] == "Male" else 1
        )
        st.markdown("---")

        st.sidebar.text_input("Chief Complaint", value=patient["complaint"])
        st.sidebar.text_area("History", value=patient["history"], height=100)
        st.markdown("### 🩺 Care Team Notes")

        st.markdown("### 📤 Upload Lab Reports")
        uploaded_files = st.file_uploader(
            "PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True
        )
        extracted = ""

    # Tabs for Interaction
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "📄 Lab Reports", "📥 Care Plan", "🕒 Timeline"]
    )

    with tab1:
        show_reasoning = st.checkbox("🧠 Show Chain of Thought")
        user_input = st.text_input("Ask a medical question")
        if user_input:
            response, reasoning = agent_response(user_input)
            st.success(response)
            if show_reasoning:
                st.markdown("### 🔍 Reasoning")
                st.code(reasoning)

    with tab2:
        uploaded_file = st.file_uploader("Upload Lab Report (PDF/Image)")
        if uploaded_file:
            parsed_results = parse_lab_pdf(uploaded_file)
            st.dataframe(parsed_results.style.apply(color_code_cells, axis=1))
            st.bar_chart(parsed_results.set_index("Test Name")["Value"])

    with tab3:
        st.subheader("📝 Care Plan")
        care_plan = generate_care_plan(patient)
        st.json(care_plan)
        if st.button("📄 Download Care Plan as PDF"):
            st.success("Care Plan exported!")

    with tab4:
        with st.expander("📆 Patient Timeline"):
            timeline = get_patient_events(patient)
            for event in timeline:
                st.markdown(f"**{event['date']}** — {event['note']}")
        with st.expander("📈 Vitals Tracker"):
            vitals_df = get_patient_vitals(patient)
            st.line_chart(
                vitals_df.set_index("Date")[
                    ["Heart Rate", "Blood Pressure", "Temperature"]
                ]
            )
            st.dataframe(vitals_df.style.apply(color_code_cells, axis=1))
        st.markdown("### 📊 Risk Scores")
        risk_scores = {
            "Sepsis": 0.85,
            "Pneumonia": 0.75,
            "COVID-19": 0.65,
            "Diabetes": 0.55,
            "Hypertension": 0.45,
        }
        st.bar_chart(risk_scores)

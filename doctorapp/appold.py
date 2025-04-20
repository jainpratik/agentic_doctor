import re

import pandas as pd
import streamlit as st
from agentic_doctor_agent import agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from multi_agent import smart_router
from PyPDF2 import PdfReader
from tools.prompts import summary_prompt
from tools.explanation_chain import explanation_chain
from tools.guideline_check import guideline_checker
from tools.medication_recommender import MedicationRecommender
from tools.procedure_explainer import procedure_chain
from tools.rag_tool import rag_tool
from tools.symptom_checker import SymptomChecker
from tools.symptom_tool import SymptomToolNeo4j
from tools.test_recommender import TestRecommender
from tools.tools import (
    SymptomTool,
    TestRecommender,
    MedicationRecommender,
    ProcedureExplainer,
    GuidelineChecker,
    LiteratureRetriever,
    LiteratureSummarizer,
    MedicationSummarizer,
)
from tools import severity_dashboard"
        
st.set_page_config(page_title="Agentic Doctor", layout="wide")
st.title("🩺 Interactive Agentic Doctor")

# Load EMR patients
with open("data/patients.json", "r") as f:
    emr_patients = json.load(f)

patient_map = {f"{p['id']} - {p['name']}": p for p in emr_patients}
selected_id = st.sidebar.selectbox("📋 Select Patient", list(patient_map.keys()))
patient = patient_map[selected_id]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "context_notes" not in st.session_state:
    st.session_state.context_notes = ""
if "summaries" not in st.session_state:
    st.session_state.summaries = []
if "entities" not in st.session_state:
    st.session_state.entities = []
if "severity_scores" not in st.session_state:
    st.session_state.severity_scores = []

def color_lab_value(value):
    try:
        val = float(value)
        if val < 0:
            return "❌"
        elif val > 1000:
            return "❌"
        elif val > 100:
            return "⚠️"
        else:
            return "✅"
    except:
        return ""


llm = ChatOpenAI(model="gpt-4", temperature=0)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

def extract_entities(text):
    pattern = r"(?P<test>[A-Za-z /]+)[:\\-\\s]+(?P<value>\\d+(\\.\\d+)?)\\s*(?P<unit>[a-zA-Z/%µ]*)"
    matches = re.findall(pattern, text)
    data = []
    scores = {}
    for m in matches:
        test = m[0].strip().upper()
        value = float(m[1])
        unit = m[3]
        flag = "✅"
        if "WBC" in test and value > 12000:
            flag = "❌"
        if "A1C" in test and value >= 6.5:
            flag = "❌"
            scores["A1C"] = value
        if "GLUCOSE" in test and value > 126:
            flag = "⚠️"
            scores["Glucose"] = value
        if "SBP" in test:
            scores["SBP"] = value
        if "RR" in test:
            scores["RR"] = value
        if "SPO2" in test:
            scores["SpO2"] = value
        if "GCS" in test:
            scores["GCS"] = value
        if "UREA" in test:
            scores["Urea"] = value
        data.append({
            "Test": test,
            "Value": value,
            "Unit": unit,
            "Flag": flag
        })
    return pd.DataFrame(data), scores


# UI layout
col1, col2 = st.columns([1, 2])


with col1:
    st.markdown("### 🧾 Patient Info")
    st.text_input("Patient ID", value=patient["id"], disabled=True)
    st.text_input("Name", value=patient["name"])
    st.number_input("Age", value=patient["age"])
    st.selectbox("Sex", ["Male", "Female"], index=0 if patient["sex"] == "Male" else 1)
    st.text_input("Chief Complaint", value=patient["complaint"])
    st.text_area("History", value=patient["history"], height=100)

    st.markdown("### 🩺 Care Team Notes")
    st.session_state.care_notes = st.text_area("Notes from physicians, nurses, etc.", height=150)

with col2:
    # Sidebar: Upload multiple files
    st.sidebar.header("📤 Upload Lab Reports")
    uploaded_files = st.sidebar.file_uploader(
        "Choose one or more files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success("✅ Files uploaded. Processing...")
        # Process each file
        # and extract text
        texts = []
        lab_scores = {}
        for file in uploaded_files:
            if file.type == "application/pdf":
                pdf = PdfReader(file)
                content = ""
                for page in pdf.pages:
                    content += page.extract_text() or ""
            else:
                content = file.read().decode("utf-8")
            texts.append(content)

        if texts:
            df, scores = extract_entities(content)
            st.session_state.entities.append(df)
            lab_scores.update(scores)
            st.session_state.context_notes = "\\n\\n".join(texts)[:4000]
            st.sidebar.success("✅ Uploaded and processed.")

            st.session_state.summaries = [summary_chain.run(text=t[:2000]) for t in texts]
            st.session_state.entities = [extract_entities(t) for t in texts]
            st.session_state.severity_scores = lab_scores
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🧪 Lab Insights", "📄 Context", "📈 Risk Scores"])

    with tab1:
        st.title("🧠 Agentic Doctor Chat")

        user_input = st.chat_input("Enter a clinical question or describe symptoms...")

        if user_input:
            full_query = (
                st.session_state.context_notes + "\\n\\nUser question: " + user_input
            )
            with st.spinner("Doctor is thinking..."):
                response = agent.run(full_query)

            st.session_state.chat_history.append(("🧑‍⚕️", user_input))
            st.session_state.chat_history.append(("🤖", response))

        for sender, msg in st.session_state.chat_history:
            st.chat_message(sender).write(msg)

    with tab2:
        st.subheader("🧬 Extracted Lab Values")
        for i, df in enumerate(st.session_state.entities):
            if not df.empty:
                st.markdown(f"**📄 File {i+1}**")
                df["Flag"] = df["Value"].apply(color_lab_value)
                st.dataframe(df)
            else:
                st.markdown(f"**📄 File {i+1}** – No structured lab values found.")

    with tab3:
        st.subheader("📑 File Summaries")
        for i, summary in enumerate(st.session_state.summaries):
            st.markdown(f"**📄 File {i+1} Summary:**")
            st.info(summary)
    with tab4:
        st.subheader("📈 Risk Score Dashboard")
        severity_dashboard.show_score_gauges(st.session_state.severity_scores)
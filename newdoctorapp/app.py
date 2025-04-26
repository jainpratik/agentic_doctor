# from pages import dashboard, login
import json
import os
from datetime import datetime
from urllib import response

import pandas as pd
import streamlit as st
from agent import generate_care_plan, get_response
from backend import color_code_cells, load_patient_data, parse_lab_pdf
from dotenv import load_dotenv
from evaluate_llm import calculate_all_metrics, evalaute_results
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from neo4j import GraphDatabase
from openai import OpenAI
from PIL import Image
from utils import color_lab_value, extract_entities, get_lab_report_path

# def main():
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False

#     if st.session_state.logged_in:
#         dashboard.run()
#     else:
#         login.run()


load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
if NEO4J_URI is None or NEO4J_USER is None or NEO4J_PASSWORD is None:
    raise ValueError("Neo4j connection details not set in environment variables.")
NEO4J_URI = NEO4J_URI.strip()
NEO4J_USER = NEO4J_USER.strip()
NEO4J_PASSWORD = NEO4J_PASSWORD.strip()
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "aid-data")
# === LLM Setup ===
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm_name = os.getenv("LLM_NAME", "gpt-4")
print(f"LLM_NAME: {llm_name}")
if llm_name not in ["gpt-3.5-turbo", "gpt-4"]:
    raise ValueError(
        "LLM_NAME environment variable must be 'gpt-3.5-turbo' or 'gpt-4'."
    )
llm_name = llm_name.strip()


def convert_graph_to_df(graph_data):
    # Extract rows
    records = []
    for section, items in graph_data.items():
        for item in items:
            records.append(
                {
                    "Type": item["type"].capitalize(),
                    "Name": item["name"],
                    "Source": item["source"],
                    "symptoms": ", ".join(item["symptoms"]),
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


# Color-coding (Streamlit Styler)
def color_metric(val, good="high"):
    color = ""
    if good == "high":
        color = "green" if val >= 0.6 else "red"
    else:  # for metrics like Perplexity (lower is better)
        color = "green" if val <= 50 else "red"
    return f"color: {color}"


def run():
    st.title("🩺 Agentic Doctor Dashboard")
    st.markdown("## Welcome to the Agentic Doctor Dashboard!")
    st.markdown("### Your AI-Powered Medical Assistant")
    st.markdown(
        "This dashboard provides a comprehensive overview of patient data, lab reports, and care plans. "
        "Interact with the AI to get insights and recommendations."
    )

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

    if "care_notes" not in st.session_state:
        st.session_state.care_notes = []

    st.markdown("---")
    # Sidebar: Patient selection
    with st.sidebar:
        st.markdown("## 🧑‍⚕️ Select Patient")
        patients = load_patient_data()
        if "selected_patient" not in st.session_state:
            st.session_state["selected_patient"] = patients[0]

        # Create name → patient map

        patient_map = {f"{p['id']} - {p['name']}": p for p in patients}

        # Sidebar patient selection
        selected_name = st.sidebar.selectbox("Select Patient", list(patient_map.keys()))

        # Get full patient object
        selected_patient = patient_map[selected_name]

        st.session_state["selected_patient"] = selected_patient

        st.markdown("---")
        st.markdown("## 📊 Patient Data")
        st.markdown("### Patient Information")
        st.markdown("**ID:** " + str(selected_patient["id"]))

        st.markdown(
            "**Name:** " + str(selected_patient["name"]),
        )
        st.markdown("**Age:** " + str(selected_patient["age"]))
        st.markdown("**Sex:** " + str(selected_patient["sex"]))
        # st.markdown("**symptoms:** " + str(selected_patient["symptoms"]))
        # st.markdown("**History:** " + str(selected_patient["history"]))

        st.text_input("symptoms", value=selected_patient["symptoms"])
        st.text_area("History", value=selected_patient["history"], height=100)

        st.markdown("### 🩺 Care Team Notes")
        st.session_state.care_notes = st.text_area(
            "Notes from physicians, nurses, etc.", height=150
        )
    # st.markdown("**Medications:** " + str(selected_patient["medications"]))
    # st.markdown("**Allergies:** " + str(selected_patient["allergies"]))
    # Tabs for Interaction
    (
        tab1,
        tab2,
        tab3,
        tab4,
        tab5,
    ) = st.tabs(
        [
            "💬 Chat",
            "📄 Lab Reports",
            "📥 Care Plan",
            "📋 Compare Results and Scoring",
            "🚀 Run Temperature Sensitivity Experiment",
        ]
    )

    with tab1:
        st.session_state.response_without_graph = None
        st.session_state.response_with_graph = None
        st.session_state.prompt_with_graph = None
        st.session_state.prompt_without = None
        st.subheader("💬 Chat with AI")
        # show_reasoning = st.checkbox("🧠 Show Chain of Thought")
        # user_input = st.text_input("Ask a medical question")
        # if user_input:
        #     response, reasoning = agent_response(user_input)
        #     st.success(response)
        #     if show_reasoning:
        #         st.markdown("### 🔍 Reasoning")
        #         st.code(reasoning)
        default_prompt = selected_patient["symptoms"]
        default_history = selected_patient["history"]
        # .split(",")
        # default_prompt = ", ".join(symptoms)
        user_input = st.text_input(
            "Enter a clinical question or describe symptoms...", value=default_prompt
        )
        user_history = st.text_input(
            "Enter a Patient history...", value=default_history
        )

        if user_input:

            with st.spinner("Doctor is thinking..."):
                # response, reasoning = agent.run(full_query)

                (
                    llm_response_with_graph,
                    llm_response_without,
                    prompt_with_graph,
                    prompt_without,
                    graph_data,
                ) = get_response(selected_patient, user_input)

                st.session_state.response_with_graph = llm_response_with_graph.content
                st.session_state.response_without_graph = llm_response_without.content
                st.session_state.prompt_with_graph = prompt_with_graph
                st.session_state.prompt_without = prompt_without

                st.session_state.chat_history.append(
                    {
                        "user": user_input,
                        "assistant": llm_response_with_graph.content,
                    }
                )
                st.markdown("### Knowledge Graph Data")
                # st.json(graph_data)
                df = convert_graph_to_df(graph_data)
                st.dataframe(df.style.apply(color_code_cells, axis=1))
                st.markdown("### Prompt with Graph")
                st.code(prompt_with_graph)

                st.markdown("### 🤖 AI Response with Graph Data")
                st.success(llm_response_with_graph.content)

                st.markdown("### Prompt without Graph")
                st.code(prompt_without)
                st.markdown("### 🤖 AI Response without graph data")
                st.success(llm_response_without.content)

            # st.session_state.chat_history.append(("🧑‍⚕️", user_input))
            # st.session_state.chat_history.append(("🤖", hypothesis))

            # if st.session_state.chat_history:
            #     st.markdown("---")
            #     st.subheader("🧠 Chat History")

            # for idx, chat in enumerate(st.session_state.chat_history):
            #     st.markdown(f"**User {idx+1}:** {chat['user']}")
            #     st.markdown(f"**Assistant {idx+1}:** {chat['assistant']}")

        for sender, msg in st.session_state.chat_history:
            st.chat_message(sender).write(msg)

    with tab2:

        st.subheader("🧪 Lab Insights")
        # Assuming you already have selected_patient
        report_path = get_lab_report_path(selected_patient["id"])
        st.markdown("### Lab Report")
        st.markdown(f"**Lab Report for Patient ID {selected_patient['id']}**")
        st.markdown(report_path)
        if report_path and os.path.exists(report_path):
            st.image(report_path, caption="Lab Report", use_column_width=True)
        else:
            st.warning("No lab report found for this patient.")

        if report_path:
            with open(report_path, "rb") as f:
                st.download_button(
                    label="📄 Download Lab Report",
                    data=f,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf",
                )
        else:
            st.warning("No lab report found for this patient.")
        st.markdown("### Lab Report")
        # parsed_results = parse_lab_pdf(open(report_path, "rb"))
        # st.dataframe(parsed_results)
        # st.markdown("### Lab Report Summary")
        # st.dataframe(parsed_results.style.apply(color_code_cells, axis=1))
        # st.bar_chart(parsed_results.set_index("Test Name")["Value"])

        # uploaded_file = st.file_uploader("Upload Lab Report (PDF/Image)")
        # if uploaded_file:
        #     parsed_results = parse_lab_pdf(uploaded_file)
        #     st.dataframe(parsed_results.style.apply(color_code_cells, axis=1))
        #     st.bar_chart(parsed_results.set_index("Test Name")["Value"])

    with tab3:
        st.subheader("📝 Care Plan")
        if st.button("📝 Generate Care Plan"):
            # Generate Care Plan
            st.session_state.response_with_graph = st.session_state.response_with_graph
            st.session_state.response_without_graph = (
                st.session_state.response_without_graph
            )
            if (
                st.session_state.response_with_graph is None
                or st.session_state.response_without_graph is None
            ):
                st.warning("Please generate a response first.")
            else:
                # Generate care plan based on the response with graph data
                selected_patient = st.session_state["selected_patient"]
                #
            analysis = st.session_state.response_with_graph
            st.markdown("### Care Plan Analysis")
            care_plan = generate_care_plan(selected_patient, analysis)
            # st.json(care_plan)
            st.markdown("### Care Plan Summary")
            st.markdown(care_plan.content)

            # st.markdown("### Care Plan Details")
            # st.dataframe(care_plan.style.apply(color_code_cells, axis=1))

            # if st.button("📄 Download Care Plan as PDF"):
            #     st.success("Care Plan exported!")

    with tab4:

        # Auto-Evaluate with LLM Self-Reflection
        if st.button("🔍 Auto-Evaluate + Score"):

            eval_response = evalaute_results(
                st.session_state.response_without_graph,
                st.session_state.response_with_graph,
            )
            # Save Auto-Evaluation

            st.subheader("🧠 LLM Self-Evaluation Result")
            st.success(eval_response.content)

            # --- Automatic Metric Scores ---

            ref = st.session_state.response_without_graph
            hyp = st.session_state.response_with_graph

            metrics = calculate_all_metrics(ref, hyp)

            # Save in session
            st.session_state["automatic_scores"] = metrics

            # Display Scores
            st.markdown("---")
            st.subheader("📊 Full Evaluation Metrics")

            st.metric("BLEU Score", f"{metrics['BLEU']:.2f}")
            st.metric("ROUGE-1 F1", f"{metrics['ROUGE-1']:.2f}")
            st.metric("ROUGE-2 F1", f"{metrics['ROUGE-2']:.2f}")
            st.metric("ROUGE-L F1", f"{metrics['ROUGE-L']:.2f}")
            st.metric("METEOR", f"{metrics['METEOR']:.2f}")
            st.metric("BERTScore Precision", f"{metrics['BERTScore Precision']:.2f}")
            st.metric("BERTScore Recall", f"{metrics['BERTScore Recall']:.2f}")
            st.metric("BERTScore F1", f"{metrics['BERTScore F1']:.2f}")
            st.metric("Perplexity", f"{metrics['Perplexity']:.2f}")
            st.metric("Exact Match", "Yes" if metrics["Exact Match"] else "No")

    with tab5:
        temperatures = [0.0, 0.2, 0.4, 0.6, 0.8]
        results = []

        if st.button("🚀 Run Full Temperature Experiment"):

            st.subheader("🚀 Running Temperature Sensitivity Experiment...")
            st.markdown(
                "This may take a few minutes. Please be patient while we run the experiment."
            )
            st.markdown("### Experiment Parameters")
            st.markdown(f"**Model:** {llm_name}")
            prompt_with_graph = st.session_state.prompt_with_graph
            prompt_without = st.session_state.prompt_without
            for temp in temperatures:
                st.write(f"Running at Temperature {temp}...")

                # Create LLM with current temp
                llm = ChatOpenAI(model="gpt-4", temperature=temp)

                # WITHOUT Knowledge Graph
                response_plain = llm.invoke(prompt_without)
                metrics_plain = calculate_all_metrics(
                    prompt_without, response_plain.content
                )

                # WITH Knowledge Graph
                response_graph = llm.invoke(prompt_with_graph)
                metrics_graph = calculate_all_metrics(
                    prompt_with_graph, response_graph.content
                )

                results.append(
                    {
                        "Temperature": temp,
                        "Mode": "Without Graph",
                        **metrics_plain,
                        "Response": response_plain.content,
                    }
                )

                results.append(
                    {
                        "Temperature": temp,
                        "Mode": "With Graph",
                        **metrics_graph,
                        "Response": response_graph.content,
                    }
                )

            # Save results
            results_df = pd.DataFrame(results)
            st.session_state["experiment_results"] = results_df

    # Display results
    if "experiment_results" in st.session_state:
        st.subheader("📊 Temperature vs Performance Table")

        styled_df = (
            st.session_state["experiment_results"]
            .style.applymap(
                color_metric,
                subset=[
                    "BLEU",
                    "ROUGE-1",
                    "ROUGE-2",
                    "ROUGE-L",
                    "METEOR",
                    "BERTScore F1",
                ],
            )
            .applymap(lambda v: color_metric(v, good="low"), subset=["Perplexity"])
        )
        st.dataframe(styled_df)

        st.subheader("📈 BLEU Score vs Temperature (By Mode)")
        st.line_chart(
            st.session_state["experiment_results"].pivot(
                index="Temperature", columns="Mode", values="BLEU"
            )
        )

        st.subheader("📈 ROUGE-L vs Temperature (By Mode)")
        st.line_chart(
            st.session_state["experiment_results"].pivot(
                index="Temperature", columns="Mode", values="ROUGE-L"
            )
        )

        st.subheader("📈 Perplexity vs Temperature (By Mode)")
        st.line_chart(
            st.session_state["experiment_results"].pivot(
                index="Temperature", columns="Mode", values="Perplexity"
            )
        )


# with st.expander("📆 Patient Timeline"):
#     timeline = get_patient_events(selected_patient)
#     for event in timeline:
#         st.markdown(f"**{event['date']}** — {event['note']}")
# with st.expander("📈 Vitals Tracker"):
#     vitals_df = get_patient_vitals(selected_patient)
#     st.line_chart(
#         vitals_df.set_index("Date")[
#             ["Heart Rate", "Blood Pressure", "Temperature"]
#         ]
#     )
#     st.dataframe(vitals_df.style.apply(color_code_cells, axis=1))
#     st.download_button(
#         "📥 Download Vitals Data", vitals_df.to_csv(), "vitals.csv", "text/csv"
#     )
#     st.success("Vitals data exported!")
#     st.markdown("---")
#     st.markdown("### 📊 Vitals Over Time")
#     st.line_chart(vitals_df.set_index("Date")[["Heart Rate", "Blood Pressure"]])
#     st.markdown("### 📈 Blood Pressure Over Time")
#     st.line_chart(vitals_df.set_index("Date")["Blood Pressure"])
#     st.markdown("### 📈 Heart Rate Over Time")
#     st.line_chart(vitals_df.set_index("Date")["Heart Rate"])
#     st.markdown("### 📈 Temperature Over Time")
#     st.line_chart(vitals_df.set_index("Date")["Temperature"])
#     st.markdown("### 📈 Weight Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Weight"])
#     # st.markdown("### 📈 BMI Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["BMI"])
#     # st.markdown("### 📈 Cholesterol Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Cholesterol"])
#     # st.markdown("### 📈 Glucose Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Glucose"])
#     # st.markdown("### 📈 Hemoglobin Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Hemoglobin"])
#     # st.markdown("### 📈 Platelets Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Platelets"])
#     # st.markdown("### 📈 WBC Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["WBC"])
#     # st.markdown("### 📈 RBC Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["RBC"])
#     # st.markdown("### 📈 Sodium Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Sodium"])
#     # st.markdown("### 📈 Potassium Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Potassium"])
#     # st.markdown("### 📈 Calcium Over Time")
#     # st.line_chart(vitals_df.set_index("Date")["Calcium"])


if __name__ == "__main__":
    run()

# agent.py
# -*- coding: utf-8 -*-
"""
Agentic Doctor - Agent Module
This module handles the interaction with the LLM and Neo4j database.
It generates Cypher queries based on patient data and retrieves relevant information from the knowledge graph.

It also provides a function to get a response from the LLM based on the selected patient.
"""

import json
import os
import re
from datetime import datetime
from tkinter import N
from webbrowser import get

from dotenv import load_dotenv
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.schema import BaseMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase
from neo4japi import (
    diagnose,
    diagnose_top,
    diagnose_top3,
    get_concepts,
    get_summary,
    query_neo4j,
)

load_dotenv()

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


# Connect to Neo4j
# Load dynamic prompt from file
def load_query_prompt():
    with open("data/prompt.txt", "r") as f:
        return f.read()


# Generate Cypher query from LLM
def generate_cypher_query(symptoms, patient_data):
    prompt_template = load_query_prompt()
    prompt = prompt_template.format(
        symptoms=symptoms,
        patient_name=patient_data["name"],
        diagnosis=patient_data["diagnosis"],
    )
    print(f"Prompt: {prompt}")
    response = llm.invoke(prompt)
    return response.content.strip()


# def split_complaints(name: str) -> List[str]:
#     # Split using commas, " and ", " or ", semicolons etc.
#     parts = re.split(r",|\band\b|\bor\b|;", name.lower())
#     # Clean up and remove tiny fragments
#     return [p.strip() for p in parts if len(p.strip()) > 2]


def get_response(selected_patient, query):
    # graph_data = query_neo4j(selected_patient["symptoms"])
    # symptoms = selected_patient["symptoms"].split(",")
    # history = selected_patient["history"].split(",")
    symptoms = selected_patient["symptoms"]
    history = selected_patient["history"]
    print(f"Symptoms: {symptoms}")
    # graph_data = get_concepts(symptoms)
    # graph_data = diagnose(symptoms, history)
    # print(f"Graph Data: {graph_data}")
    graph_data = diagnose_top3(symptoms, history)
    print(f"Top 3 Diagnoses: {graph_data}")
    # top_Diagnosis = diagnose_top(symptoms, history)
    # print(f"Top Diagnosis: {top_Diagnosis}")
    # graph_data = query_neo4j(selected_patient["symptoms"])

    if not graph_data:
        print("No relevant data found in the knowledge graph.")

    prompt_template_with_graph = """
        You are a clinical decision support AI.

        Based on the following patient's profile and associated medical knowledge graph, your task is to:
        Analyze the patient's sex, age, symptoms, and medical history.
        Examine the provided possible diseases, linked symptoms, and linked history.
        Evaluate the suggested diagnostic tests and procedures.

        ⚡ Please reason step-by-step to:
        Infer the top 3 most probable diagnoses for this patient, ordered by likelihood.
        Suggest the most appropriate diagnostic tests to confirm or rule out these diagnoses.
        Recommend first-line treatments based on standard medical guidelines.
        If multiple diagnoses are possible, explain differential considerations.

        🛡️ Important:
        Be cautious.
        If information is insufficient for a definitive diagnosis, say so clearly.
        Only reason based on the given structured data (do not hallucinate external facts).
        Here are the patient details:
                Name: {name}
                Age: {age}
                Gender: {gender}
                Symptoms:{symptoms}
                History:{history}
        📦 Here is the structured patient knowledge graph:
            {graph_data}
    """

    # Assuming you have selected_patient already
    # prompt_template_with_graph = """
    #             You are a medical assistant.
    #             Given the following patient details:

    #             Name: {name}
    #             Age: {age}
    #             Gender: {gender}
    #             Symptoms:{symptoms}
    #             History:{history}

    #             Relevant Knowledge Graph Data:
    #             {graph_data}

    #             Generate an informed  recommendation, procedures, tests and next steps with chain of thought and reasoning based on the above.
    #             """

    prompt_template_without = """
                You are a medical assistant.
                Given the following patient details:

                Name: {name}
                Age: {age}
                Gender: {gender}
                Symptoms:{symptoms}
                History:{history}

                Identify a primary diagnosis and generate an informed recommendation, procedures, tests and next steps with chain of thought and reasoning based on the above.
                """

    # custom_instruction = st.text_area("Add your custom prompt or instruction here")

    # if st.button("🧠 Generate Response from LLM"):
    #     if custom_instruction:
    prompt_with_graph = prompt_template_with_graph.format(
        name=selected_patient["name"],
        age=selected_patient["age"],
        gender=selected_patient["sex"],
        symptoms=selected_patient["symptoms"],
        history=selected_patient["history"],
        graph_data=graph_data,
    )

    prompt_without = prompt_template_without.format(
        name=selected_patient["name"],
        age=selected_patient["age"],
        gender=selected_patient["sex"],
        symptoms=selected_patient["symptoms"],
        history=selected_patient["history"],
    )

    print(f"prompt_with_graph: {prompt_with_graph}")
    # Call OpenAI

    # LLM setup
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    response_with_graph = llm.invoke(prompt_with_graph)
    response_without = llm.invoke(prompt_without)
    # response = "TESTING WITHOUT LLM"
    print(response_with_graph)
    print(response_without)

    return (
        response_with_graph,
        response_without,
        prompt_with_graph,
        prompt_without,
        graph_data,
    )


# def get_response(selected_patient):

#     patients = load_patient_data()
#     patient_data = next(p for p in patients if p["name"] == selected_patient)
#     symptoms = patient_data["symptoms"]
#     notes = patient_data["notes"]
#     cypher_query = generate_cypher_query(symptoms, patient_data)
#     print(f"Generated Cypher Query: {cypher_query}")
#     results = query_neo4j(notes)
#     return results, cypher_query


def generate_care_plan(selected_patient, analysis):
    """
    Generate a care plan based on the selected patient's data and knowledge graph context.
    """
    analysis = analysis.strip()
    careplan_prompt = """
    You are a highly experienced clinical assistant helping create evidence-based care plans.

    Given the following patient details:

    Patient Name: {name}
    Age: {age}
    Gender: {gender}
    Symptoms: {symptoms}
    History: {history}

    Additional Information:
    {graph_data}
    
    Initial Analysis:{analysis}

    Talk about the patient's condition and generate a detailed care plan.
    The care plan should include the following sections:
     Primary Diagnosis:
    - Include the most likely diagnosis based on the symptoms and history provided.
    - Discuss the reasoning behind this diagnosis, including any relevant lab results or imaging findings.
    - Mention any differential diagnoses considered and why they were ruled out.
    - Include any relevant knowledge graph data that supports the diagnosis.
    
    Please generate a detailed care plan following this format:

    1. **Summary of Patient's Condition** (in 2-3 sentences)
    2. **Immediate Management Recommendations** (tests, monitoring, urgent actions)
    3. **Medications or Treatments** (if applicable, with general classes not specific brand names)
    4. **Follow-up Actions** (timelines for re-assessment, labs, imaging)
    5. **Patient Education Points** (what patient should know/do at home)
    6. **Red Flags to Watch For** (warning symptoms that require urgent care)

    Instructions:
    - Base the care plan on best clinical practices.
    - Prioritize safety and clarity.
    - Avoid recommending specific prescriptions unless absolutely necessary.
    - Keep explanations patient-friendly but medically accurate.
    - Structure the response using numbered or bulleted points where appropriate.
    """
    symptoms = selected_patient["symptoms"]
    history = selected_patient["history"]

    graph_data = build_patient_knowledge_graph(selected_patient)
    print(f"Knowledge Graph: {graph_data}")

    # Fill in the prompt with patient data and knowledge graph context
    careplan_prompt = careplan_prompt.format(
        name=selected_patient["name"],
        age=selected_patient["age"],
        gender=selected_patient["sex"],
        symptoms=selected_patient["symptoms"],
        history=selected_patient["history"],
        graph_data=graph_data,
        analysis=analysis,
    )

    llm = ChatOpenAI(model="gpt-4", temperature=0.3)

    response_with_graph = llm.invoke(careplan_prompt)
    # response = "TESTING WITHOUT LLM"
    print(response_with_graph)

    return response_with_graph

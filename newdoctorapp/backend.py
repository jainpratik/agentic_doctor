import json
import os
from datetime import datetime
from json import load

import pandas as pd
import pdf2image
import pytesseract
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from neo4j import GraphDatabase
from openai import OpenAI
from PIL import Image

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm_name = os.getenv("LLM_NAME", "gpt-4")
print(f"LLM_NAME: {llm_name}")
if llm_name not in ["gpt-3.5-turbo", "gpt-4"]:
    raise ValueError(
        "LLM_NAME environment variable must be 'gpt-3.5-turbo' or 'gpt-4'."
    )

llm = ChatOpenAI(temperature=0, model_name="gpt-4")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_password"))


def load_patient_data():
    with open("data/patients.json") as f:
        return json.load(f)


def get_patient_info(name):
    patients = load_patient_data()
    return next(p for p in patients if p["name"] == name)


def agent_response(query):
    tools = []
    agent = initialize_agent(tools, llm, verbose=True)
    response = agent.run(query)
    return response, "Chain of reasoning (simulated)."


def query_neo4j(disease):
    with driver.session() as session:
        result = session.run(
            "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) WHERE d.name = $name RETURN s.name",
            name=disease,
        )
        return [record["s.name"] for record in result]


def parse_lab_pdf(uploaded_file):
    images = pdf2image.convert_from_bytes(uploaded_file.read())
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return extract_lab_data(text)


def extract_lab_data(text):
    # Simplified parsing logic
    import re

    results = []
    for line in text.split("\n"):
        match = re.match(r"(.+?)\s+([\d.]+)\s+\((.+?)\)", line)
        if match:
            results.append(
                {
                    "Test Name": match[1],
                    "Value": float(match[2]),
                    "Reference Range": match[3],
                }
            )
    return pd.DataFrame(results)


# def parse_lab_pdf(uploaded_file):
#     images = pdf2image.convert_from_bytes(uploaded_file.read())
#     text = "\n".join([pytesseract.image_to_string(img) for img in images])
#     results = []
#     for line in text.split("\\n"):
#         parts = line.split()
#         if len(parts) >= 3:
#             try:
#                 results.append(
#                     {
#                         "Test Name": parts[0],
#                         "Value": float(parts[1]),
#                         "Reference Range": " ".join(parts[2:]),
#                     }
#                 )
#             except:
#                 continue
#     return pd.DataFrame(results)


def color_code_cells(row):
    if "Value" in row and isinstance(row["Value"], (float, int)):
        if row["Value"] > 140:
            return ["background-color: yellow"] * len(row)
    return [""] * len(row)


def generate_care_plan(patient):
    return {
        "patient": patient["name"],
        # "diagnosis": patient["diagnosis"],
        "recommendations": [
            "Follow-up in 2 weeks",
            "Routine labs and imaging",
            "Lifestyle and diet changes",
        ],
    }


# def generate_pdf(care_plan):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt=f"Care Plan for {care_plan['patient']}", ln=True)
#     pdf.cell(200, 10, txt=f"Diagnosis: {care_plan['diagnosis']}", ln=True)
#     for rec in care_plan["recommendations"]:
#         pdf.cell(200, 10, txt=f"- {rec}", ln=True)
#     pdf.output("care_plan.pdf")


def get_patient_events(patient):
    return [
        {"date": "2024-04-01", "note": "Routine check-up"},
        {"date": "2024-04-15", "note": "Complained of chest pain"},
        {"date": "2024-04-20", "note": "Lab tests done"},
    ]


def get_patient_vitals(patient):
    return pd.DataFrame(
        {
            "Date": ["2024-04-01", "2024-04-15", "2024-04-20"],
            "Heart Rate": [72, 78, 85],
            "Blood Pressure": [120, 130, 135],
            "Temperature": [98.6, 99.1, 99.3],
        }
    )

import json
import os
import re
from datetime import datetime
from tkinter import N

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


def serialize_state(state):
    def convert_value(val):
        if isinstance(val, BaseMessage):
            return val.dict()
        elif isinstance(val, list):
            return [convert_value(v) for v in val]
        elif isinstance(val, dict):
            return {k: convert_value(v) for k, v in val.items()}
        else:
            return val

    return convert_value(state)


# === Logging helper ===
def log_state_change(patient_id, step, state):
    """
    Logs the state change of a patient during a specific step of a process.

    This function records the state changes of a patient into a JSON file
    located in the "state_logs" directory. If the log file for the patient
    does not exist, it creates a new one. Each state change is appended to
    the log with a timestamp.

    Args:
        patient_id (str): The unique identifier for the patient.
        step (str): The current step or phase of the process.
        state (dict): The state information to be logged.

    Raises:
        OSError: If there is an issue creating the "state_logs" directory
                 or writing to the log file.
    """
    os.makedirs("state_logs", exist_ok=True)
    path = f"state_logs/{patient_id}_state_log.json"

    # if not os.path.exists(path):
    #     with open(path, "w") as f:
    #         json.dump(serialize_state(history), f, indent=2)

    with open(path, "r") as f:
        history = json.load(f)
    history.append(
        {"step": step, "timestamp": datetime.now().isoformat(), "state": state}
    )
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# Step 2: Query Neo4j for support
class Neo4jReasoner:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            print(f"Query result: {result}")
            return [record for record in result]
        

    def match_diseases(self, symptoms):
        query = """
        UNWIND $symptoms AS symptom
        MATCH (s:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)
        WHERE toLower(s.name) = toLower(symptom)
        RETURN d.name AS disease, count(*) AS support
        ORDER BY support DESC
        LIMIT 3
        """
        with self.driver.session() as session:
            result = session.run(query, symptoms=symptoms)
            print(f"Query result: {result}")
            if not result:
                result = []
            else:
                result = [
                    {"disease": r["disease"], "support": r["support"]} for r in result
                ]
                print(f"Formatted result: {result}")
            return result

    def query_knowledge_graph(self, query):
        cypher_query = f"""
        MATCH (d:Disease)-[:CAUSES]->(s:Symptom)
        WHERE d.name CONTAINS '{query}'
        RETURN d.name AS disease, s.name AS symptom
        """
        result = self.query(cypher_query)
        print(f"Query result: {result}")
        if not result:
            return "No results found."
        else:
            result = [
                f"{record['disease']} causes {record['symptom']}" for record in result
            ]
            result = "\n".join(result)
            print(f"Formatted result: {result}")
        return result

    def get_tests_for_disease(self, disease):
        query = """
        MATCH (d:Disease {name: $disease})-[:HAS_TEST]->(t:Test)
        RETURN t.name AS test
        """
        with self.driver.session() as session:
            results = session.run(query, disease=disease)
            tests = [r["Suggested tests"] for r in results]

            print(f"Suggested tests: {tests}")
            if not tests:
                tests = {"Suggested tests": ["Basic metabolic panel"]}
            else:
                tests = [f"Suggested tests: {test}" for test in tests]
                print(f"Formatted tests: {tests}")

        return tests


# Step 3: Final response formatter
def format_structured_response(
    diagnosis, supported_symptoms, suggested_tests, rule_outs, next_steps, follow_ups
):
    return {
        "diagnosis": diagnosis,
        "symptom_support": supported_symptoms,
        "tests": suggested_tests,
        "rule_out": rule_outs,
        "next_steps": next_steps,
        "follow_ups": follow_ups,
    }


# Step 4: Mermaid diagram
def generate_mermaid_trace():
    return """
    <pre class="mermaid">
    graph TD
        A[📝 Input Notes] --> B[🧠 LLM Hypothesis]
        B --> C[🔍 KG Query: Symptom → Disease]
        C --> D[📋 Guideline Check]
        D --> E[💡 Final Diagnosis]
    </pre>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true });
    </script>
    """


def get_memory(session_id):
    return FileChatMessageHistory(f"history/{session_id}.json")


def get_hypothesis_chain():
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    # Step 1: Generate hypothesis
    hypothesis_prompt = PromptTemplate(
        input_variables=["notes"],
        template="""You are a diagnostic assistant. Given the clinical notes:{notes}
    List top 2 possible diagnoses with justification and any missing data you’d like to ask the patient about.
    """,
    )
    hypothesis_chain = RunnableWithMessageHistory(
        hypothesis_prompt | llm,
        get_session_history=get_memory,
        input_messages_key="notes",
        history_messages_key="chat_history",
    )
    return hypothesis_chain


def get_symptom_chain():
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    symptom_extract_prompt = PromptTemplate(
        input_variables=["notes"],
        template="Extract a list of symptoms from this clinical text:\n\n{notes}\n\nList only the symptoms.",
    )
    symptom_chain = RunnableWithMessageHistory(
        symptom_extract_prompt | llm,
        get_session_history=get_memory,
        input_messages_key="notes",
        history_messages_key="chat_history",
    )
    return symptom_chain


# === Graph Builder with Neo4j ===
def create_agentic_doctor_graph_with_neo4j():
    kg = Neo4jReasoner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    workflow = StateGraph()

    workflow.add_state_keys(
        ["notes", "symptoms", "hypothesis", "graph_diseases", "tests", "patient_id"]
    )

    def extract_symptoms_auto(state):
        notes = state["notes"]
        regex_hits = re.findall(
            r"\\b(?:fever|cough|pain|fatigue|headache|nausea|vomiting|shortness of breath)\\b",
            notes.lower(),
        )
        unique = list(set(regex_hits))
        if unique:
            result = {"symptoms": unique}
        else:
            llm_result = generate_symptoms(state)
            print(f"LLM result: {llm_result}")
            if not llm_result:
                extracted = [
                    s.strip() for s in re.split(r"[\\n,;]", llm_result) if s.strip()
                ]
                llm_result = {"symptoms": extracted}

            print(f"symptoms: {llm_result}")

        log_state_change(
            state.get("patient_id", "unknown"), "ExtractSymptoms", llm_result
        )
        return llm_result

    # Extract symptoms (assume comma-separated string in `notes`)
    # def extract_symptoms(state):
    # symptoms = [s.strip() for s in state["notes"].lower().split(",") if s.strip()]
    # if not symptoms:
    #     raise ValueError("No symptoms found in notes.")
    # print(f"Extracted symptoms: {symptoms}")
    # return {"symptoms": symptoms}

    # Use LLM to hypothesize
    def run_hypothesis_llm(state):
        result = generate_hypothesis(state)
        print(f"Generated hypothesis: {result}")
        log_state_change(
            state.get("patient_id", "unknown"), "GenerateHypothesis", result
        )
        return result

    # Query Neo4j using symptoms
    def query_neo4j(state):
        matches = kg.match_diseases(state["symptoms"])
        print(f"Matched diseases: {matches}")
        if not matches:
            return {"graph_diseases": ["No diseases found"]}
        else:
            # Format the results for better readability
            matches = [f"Possible disease: {d}" for d in matches]
            print(f"Formatted matches: {matches}")
        return {"graph_diseases": matches}

    # Suggest tests based on disease match
    def suggest_tests_from_graph(state):
        disease = state["graph_diseases"][0] if state["graph_diseases"] else None
        if not disease:
            return {"Suggested tests": []}
        tests = kg.get_tests_for_disease(disease)
        print(f"Suggested tests: {tests}")
        return tests

    workflow.add_node("ExtractSymptoms", RunnableLambda(extract_symptoms_auto))
    workflow.add_node("GenerateHypothesis", RunnableLambda(run_hypothesis_llm))
    workflow.add_node("QueryNeo4jUMLS", RunnableLambda(query_neo4j))
    workflow.add_node("SuggestTests", RunnableLambda(suggest_tests_from_graph))

    workflow.set_entry_point("ExtractSymptoms")
    workflow.add_edge("ExtractSymptoms", "GenerateHypothesis")
    workflow.add_edge("GenerateHypothesis", "QueryNeo4jUMLS")
    workflow.add_edge("QueryNeo4jUMLS", "SuggestTests")
    workflow.add_edge("SuggestTests", END)

    return workflow.compile()


def run_langgraph_pipeline(notes):
    symptoms = [s.strip() for s in notes.split(",") if s.strip()]
    graph = create_agentic_doctor_graph_with_neo4j()
    result = graph.invoke({"notes": notes, "symptoms": symptoms})
    return result


def generate_hypothesis(state):
    session_id = state.get("patient_id", "unknown")
    result = get_hypothesis_chain().invoke(
        {"notes": state["notes"]}, config={"configurable": {"session_id": session_id}}
    )
    out = {"hypothesis": result.content}
    log_state_change(session_id, "GenerateHypothesis", out)
    return out


def generate_symptoms(state):
    session_id = state.get("patient_id", "unknown")
    result = get_symptom_chain().invoke(
        {"notes": state["notes"]}, config={"configurable": {"session_id": session_id}}
    )
    out = {"symptoms": result.content}
    log_state_change(session_id, "GenerateSymptoms", out)
    return out


# Step 4: Run the pipeline
def run_reasoning_pipeline(notes, symptoms, patient_id):

    hypo_text = get_hypothesis_chain().invoke(
        {"notes": notes}, config={"configurable": {"session_id": patient_id}}
    )

    kg = Neo4jReasoner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    support = kg.match_diseases(symptoms)

    top_diagnosis = support[0]["disease"] if support else "Unknown"
    suggested_tests = (
        ["Chest X-ray", "CBC"]
        if "pneumonia" in top_diagnosis.lower()
        else ["Further imaging"]
    )
    rule_outs = ["COVID-19"] if "fever" in [s.lower() for s in symptoms] else []
    next_steps = ["Order " + test for test in suggested_tests]
    follow_ups = ["When did symptoms start?", "Recent travel or contact exposure?"]

    structured = format_structured_response(
        diagnosis=top_diagnosis,
        supported_symptoms=symptoms,
        suggested_tests=suggested_tests,
        rule_outs=rule_outs,
        next_steps=next_steps,
        follow_ups=follow_ups,
    )

    # Save reasoning trace to logs
    os.makedirs("reasoning_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"reasoning_logs/{patient_id}_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "patient_id": patient_id,
                "timestamp": timestamp,
                "input_notes": notes,
                "symptoms": symptoms,
                "hypothesis": hypo_text,
                "structured": structured,
            },
            f,
            indent=2,
        )

    return hypo_text, structured, generate_mermaid_trace()


# # Initialize the LLM chain for hypothesis generation
# notes = "Patient has a fever and cough."
# symptoms = ["fever", "cough"]
# hypothesis, structured, mermaid_trace = run_reasoning_pipeline(
#     notes, symptoms, "test_patient"
# )
# print("Input Notes:", notes)
# print("Hypothesis:", hypothesis)
# print("Structured Output:", structured)
# print("Mermaid Trace:", mermaid_trace)

# result = run_langgraph_pipeline(notes)
# print("LangGraph Result:", result)

import os
from datetime import datetime
from json import load

from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.memory import Memory
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase

# # Load the configuration file
# config_path = os.path.join(os.path.dirname(__file__), "config.json")
# with open(config_path, "r") as config_file:
#     config = load(config_file)
#     print(f"Config loaded: {config}")

#     # Extract the UMLS API key from the config
#     umls_api_key = config.get("umls_api_key")
#     if not umls_api_key:
#         raise ValueError("UMLS API key not found in the configuration file.")
#     print(f"UMLS API key: {umls_api_key}")


# === Neo4j UMLS connection ===
class UMLSReasoner:

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
        UNWIND $symptoms AS s
        MATCH (sym:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)
        WHERE toLower(sym.name) = toLower(s)
        RETURN d.name AS disease, count(*) AS support
        ORDER BY support DESC LIMIT 3
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


# === Graph Builder with Neo4j ===
def create_agentic_doctor_graph_with_neo4j():
    kg = UMLSReasoner("bolt://localhost:7687", "neo4j", "password")

    workflow = StateGraph()
    workflow.add_state_keys(
        ["notes", "symptoms", "hypothesis", "graph_diseases", "tests"]
    )

    # Extract symptoms (assume comma-separated string in `notes`)
    def extract_symptoms(state):
        symptoms = [s.strip() for s in state["notes"].lower().split(",") if s.strip()]
        if not symptoms:
            raise ValueError("No symptoms found in notes.")
        print(f"Extracted symptoms: {symptoms}")
        return {"symptoms": symptoms}

    # Use LLM to hypothesize
    def run_llm(state):
        return {"hypothesis": hypothesis_chain.run(notes=state["notes"])}

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
    def test_suggester(state):
        if any("pneumonia" in d.lower() for d in state["graph_diseases"]):
            return {"tests": ["Chest X-ray", "CBC"]}
        else:
            return {"tests": ["Basic metabolic panel"]}

    workflow.add_node("ExtractSymptoms", RunnableLambda(extract_symptoms))
    workflow.add_node("GenerateHypothesis", RunnableLambda(run_llm))
    workflow.add_node("QueryNeo4jUMLS", RunnableLambda(query_neo4j))
    workflow.add_node("SuggestTests", RunnableLambda(test_suggester))

    workflow.set_entry_point("ExtractSymptoms")
    workflow.add_edge("ExtractSymptoms", "GenerateHypothesis")
    workflow.add_edge("GenerateHypothesis", "QueryNeo4jUMLS")
    workflow.add_edge("QueryNeo4jUMLS", "SuggestTests")
    workflow.add_edge("SuggestTests", END)

    return workflow.compile()


# Generate base LangGraph DAG for Agentic Doctor reasoning


# Step 5: Run the pipeline
def run_reasoning_pipeline(notes, symptoms, patient_id="unknown"):
    hypo_text = hypothesis_chain.run(notes=notes)
    kg = Neo4jReasoner("bolt://localhost:7687", "neo4j", "password")
    support = kg.get_supported_diseases(symptoms)

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


# @tool
# def check_guidelines(symptoms: list) -> str:
#     if "fever" in symptoms and "shortness of breath" in symptoms:
#         return "⚠️ Sepsis possible. Suggest qSOFA check."
#     return "No high-risk guideline match."


# Example query


# client = ChatOpenAI(api_key=openai_key, model=llm_name)
# client = ChatOpenAI(api_key=openai_key, model=llm_name)
# client = OpenAI(api_key=openai_key)
# client = OpenAI(
# Initialize LLM (replace with BioGPT or ClinicalBERT endpoint)
# llm = OpenAI(model="gpt-4", temperature=0)

# # Define memory
# memory = ConversationBufferMemory(memory_key="chat_history")

# # Define tools (link to knowledge graph and RAG)
# tools = [
#     Tool(
#         name="Knowledge Graph",
#         func=lambda query: query_knowledge_graph(query),  # Placeholder function
#         description="Use for structured medical information.",
#     ),
#     # Tool(
#     #     name="Research Papers",
#     #     func=lambda query: retrieve_from_research(query),  # Placeholder function
#     #     description="Use for medical literature.",
#     # ),
# ]

# # Initialize Agent
# agent = initialize_agent(
#     tools, client, agent="conversational-react-description", memory=memory, verbose=True
# )

# # Test the agent
# response = agent.run("What are the common causes of chest pain?")
# print(response)


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


# === LLM and Chains ===
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

hypothesis_prompt = PromptTemplate(
    input_variables=["notes"],
    template="Based on the following clinical notes, hypothesize up to two possible diagnoses with justification:{notes}",
)
hypothesis_chain = LLMChain(llm=llm, prompt=hypothesis_prompt)

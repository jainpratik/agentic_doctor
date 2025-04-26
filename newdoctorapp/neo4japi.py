import json
import os
import re
from typing import Dict, List

# import uvicorn
from dotenv import load_dotenv

# from fastapi import FastAPI, Query
from neo4j import GraphDatabase

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "aid_2025")
if NEO4J_URI is None or NEO4J_USER is None or NEO4J_PASSWORD is None:
    raise ValueError("Neo4j connection details not set in environment variables.")
NEO4J_URI = NEO4J_URI.strip()
NEO4J_USER = NEO4J_USER.strip()
NEO4J_PASSWORD = NEO4J_PASSWORD.strip()
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "aid-data")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---- INIT ----
# app = FastAPI(
#     title="Biomedical Concepts API",
#     description="Query UMLS-based Neo4j biomedical concepts dynamically, grouped by type (disorder, procedure, test, anatomy, etc).",
#     version="1.0.2",
# )
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Query Neo4j database
def query_neo4j(notes):
    # cypher_query = f"""
    #     MATCH (c:Concept)
    #     WHERE tolower(c.name) CONTAINS '{notes}' AND ('procedures' in labels(c))
    #     RETURN c
    #     LIMIT 25
    #     """
    cypher_query = (
        f"""MATCH (c:Concept) WHERE c.name contains '{notes}' return c limit 25 """
    )
    print(f"Cypher Query: {cypher_query}")

    with driver.session() as session:
        result = session.run(cypher_query)
        print(f"Result: {result}")
        if not result:
            print("No relevant data found in the knowledge graph.")
            return None
        return [record.data() for record in result]


# ---- UTILITY FUNCTIONS ----
def remove_concept_label(labels: List[str]) -> List[str]:
    return [label for label in labels if label != "Concept"]


def get_primary_label(labels: List[str], name: str) -> str:
    semantic_labels = [
        "disorders",
        "procedures",
        "physiology",
        "anatomy",
        "chemicals_drugs",
        "living_beings",
        "devices",
        "occupations",
        "phenomena",
        "organizations",
        "activities_behaviors",
        "geographic_areas",
        "genes_molecular_sequences",
    ]

    if "procedures" in labels and any(
        keyword in name.lower()
        for keyword in [
            "test",
            "x-ray",
            "mri",
            "ct",
            "scan",
            "imaging",
            "lab",
            "blood",
            "analysis",
            "screening",
        ]
    ):
        return "tests"
    for label in semantic_labels:
        if label in labels:
            return label
    return "other"


def map_type(label: str) -> str:
    mapping = {
        "disorders": "disorder",
        "procedures": "procedure",
        "tests": "test",
        "physiology": "physiology",
        "anatomy": "anatomy",
        "chemicals_drugs": "chemical_drug",
        "living_beings": "living_being",
        "devices": "device",
        "occupations": "occupation",
        "phenomena": "phenomenon",
        "organizations": "organization",
        "activities_behaviors": "activity_behavior",
        "geographic_areas": "geographic_area",
        "genes_molecular_sequences": "gene_or_sequence",
        # "other": "other",
    }
    return mapping.get(label, "other")


def split_symptoms(name: str) -> List[str]:
    # Split using commas, " and ", " or ", semicolons etc.
    parts = re.split(r",|\band\b|\bor\b|;", name.lower())
    # Clean up and remove tiny fragments
    return [p.strip() for p in parts if len(p.strip()) > 2]


# class DiagnoseRequest(BaseModel):
#     symptoms: List[str]
#     history: List[str] = []
def diagnose_top(symptoms: List[str], history: List[str]) -> Dict[str, List[Dict]]:
    """
    Find the top probable diagnosis given symptoms and patient history.
    Uses fuzzy search + scoring model.
    """
    # symptoms = payload.symptoms
    # history = payload.history

    session = driver.session()

    results = []
    for symptom in symptoms:
        query = """
        CALL db.index.fulltext.queryNodes("conceptNameIndex", $symptomQuery) YIELD node, score
        RETURN node.cui AS cui, node.name AS name, node.source AS source, labels(node) AS all_labels, score
        LIMIT 20
        """
        matches = session.run(query, symptomQuery=symptom)
        results.extend(matches.data())

    session.close()

    # Merge and deduplicate
    seen = {}
    for concept in results:
        name = concept["name"]
        if name not in seen:
            seen[name] = {
                "cui": concept["cui"],
                "name": name,
                "source": concept["source"],
                "labels": concept["all_labels"],
                "fuzzy_score": concept["score"],
            }
        else:
            # Merge scores if seen multiple times
            seen[name]["fuzzy_score"] += concept["score"]

    # Score each concept
    concepts_scored = []
    for item in seen.values():
        labels = remove_concept_label(item["labels"])
        primary_label = get_primary_label(labels, item["name"])
        symptoms = split_symptoms(item["name"])
        concept_name_lower = item["name"].lower()

        symptom_score = sum(1 for s in symptoms if s.lower() in concept_name_lower)
        history_score = sum(1 for h in history if h.lower() in concept_name_lower)

        total_score = (
            (symptom_score * 10) + (history_score * 5) + (item["fuzzy_score"] * 5)
        )

        concepts_scored.append(
            {
                "type": map_type(primary_label),
                "name": item["name"],
                "source": item["source"],
                "symptoms": symptoms,
                "score": total_score,
            }
        )

    # Only keep disorders
    disorders = [c for c in concepts_scored if c["type"] == "disorder"]

    if not disorders:
        return {"message": "No probable disorder found"}

    # Sort by score descending
    disorders = sorted(disorders, key=lambda x: x["score"], reverse=True)

    # Return top 1
    return disorders[0]


def diagnose_top3(symptoms: List[str], history: List[str]) -> Dict[str, List[Dict]]:
    """
    Find top 3 most probable concepts per type (disorder, test, procedure),
    using fuzzy search + symptoms + history scoring.
    """
    session = driver.session()

    results = []
    for symptom in symptoms:
        # query = """
        # CALL db.index.fulltext.queryNodes("conceptNameIndex", $symptomQuery) YIELD node, score
        # RETURN node.cui AS cui, node.name AS name, node.source AS source, labels(node) AS all_labels, score
        # LIMIT 20
        # """
        query = """
        CALL db.index.fulltext.queryNodes("conceptNameIndex", $symptomQuery) YIELD node, score
        RETURN node.cui AS cui, node.name AS name, node.source AS source, labels(node) AS all_labels, score
        LIMIT 20
        """
        matches = session.run(query, symptomQuery=symptom)
        results.extend(matches.data())

    session.close()

    # Merge and deduplicate
    seen = {}
    for concept in results:
        name = concept["name"]
        if name not in seen:
            seen[name] = {
                "cui": concept["cui"],
                "name": name,
                "source": concept["source"],
                "labels": concept["all_labels"],
                "fuzzy_score": concept["score"],
            }
        else:
            # Merge fuzzy scores if multiple matches
            seen[name]["fuzzy_score"] += concept["score"]

    # Score each concept
    concepts_scored = []
    for item in seen.values():
        labels = remove_concept_label(item["labels"])
        primary_label = get_primary_label(labels, item["name"])
        symptoms = split_symptoms(item["name"])
        concept_name_lower = item["name"].lower()

        symptom_score = sum(1 for s in symptoms if s.lower() in concept_name_lower)
        history_score = sum(1 for h in history if h.lower() in concept_name_lower)

        total_score = (
            (symptom_score * 10) + (history_score * 5) + (item["fuzzy_score"] * 5)
        )

        concepts_scored.append(
            {
                "type": map_type(primary_label),
                "name": item["name"],
                "source": item["source"],
                "symptoms": symptoms,
                "score": total_score,
            }
        )

    # Group by type
    grouped: Dict[str, List[Dict]] = {}
    for concept in concepts_scored:
        concept_type = concept["type"]
        if concept_type not in grouped:
            grouped[concept_type] = []
        grouped[concept_type].append(concept)

    # Sort each group by score and take top 3
    top3_grouped: Dict[str, List[Dict]] = {}
    for concept_type, concepts in grouped.items():
        sorted_concepts = sorted(concepts, key=lambda x: x["score"], reverse=True)
        top3_grouped[concept_type] = sorted_concepts[:3]  # Top 3 only

    return top3_grouped


def diagnose(symptoms: List[str], history: List[str]) -> Dict[str, List[Dict]]:
    """
    Given symptoms and patient history, find likely disorders, tests, procedures.
    Narrows down using both current symptoms and medical history.
    """
    # symptoms = payload.symptoms
    # history = payload.history

    # Smart Cypher: match symptoms, narrow by history
    # query = """
    # MATCH (c:Concept)
    # WHERE ANY(symptom IN $symptoms WHERE toLower(c.name) CONTAINS toLower(symptom))
    # RETURN c.cui AS cui, c.name AS name, c.source AS source, labels(c) AS all_labels
    # LIMIT 100
    # """
    query = """
    MATCH (c:Concept)
    WHERE ANY(symptom IN $symptoms WHERE toLower(c.name) CONTAINS toLower(symptom))
    RETURN c.cui AS cui, c.name AS name, c.source AS source, labels(c) AS all_labels
    LIMIT 100
    """
    score_threshold = 3  # minimum score required to keep a result

    session = driver.session()
    result = session.run(query, symptoms=symptoms)
    concepts = result.data()
    print(f"Concepts: {concepts}")
    session.close()

    grouped: Dict[str, List[Dict]] = {}

    for concept in concepts:
        labels = remove_concept_label(concept["all_labels"])
        print(f"Labels: {labels}")
        primary_label = get_primary_label(labels, concept["name"])
        print(f"Primary Label: {primary_label}")
        # # Check if the concept is a test or procedure
        # if primary_label == "tests" or primary_label == "procedures":
        #     # Check if the concept name contains any of the symptoms
        #     if not any(
        #         symptom in concept["name"].lower()
        #         for symptom in [s.lower() for s in symptoms]
        #     ):
        #         continue
        symptoms = split_symptoms(concept["name"])
        print(f"symptoms: {symptoms}")
        concept_name_lower = concept["name"].lower()
        print(f"Concept Name Lower: {concept_name_lower}")
        # # Narrowing logic based on history
        # if history:
        #     # Boost if history matches (symptoms overlap with history terms)
        #     if not any(
        #         h in concept["name"].lower() for h in [hx.lower() for hx in history]
        #     ):
        #         continue  # SKIP concept if no history relevance

        # --- NEW: Scoring instead of Hard Filter ---
        # history_match = (
        #     any(h in concept["name"].lower() for h in [hx.lower() for hx in history])
        #     if history
        #     else False
        # )
        # print(f"History Match: {history_match}")

        # --- Scoring ---
        symptom_score = sum(1 for s in symptoms if s.lower() in concept_name_lower)
        history_score = sum(1 for h in history if h.lower() in concept_name_lower)

        total_score = (symptom_score * 10) + (history_score * 5)
        history_match = history_score > 0

        # --- Filter by score ---
        if total_score < score_threshold:
            continue  # Skip concepts with very low relevance

        concept_entry = {
            "type": map_type(primary_label),
            "name": concept["name"],
            "source": concept["source"],
            "symptoms": symptoms,
            "history_match": history_match,
            "score": total_score,
        }

        if primary_label not in grouped:
            grouped[primary_label] = []
        grouped[primary_label].append(concept_entry)

    return grouped


# ---- API ROUTES ----
# @app.get("/concepts", response_model=Dict[str, List[Dict]], tags=["Concept Search"])
def get_concepts(
    symptoms: List[str],
    # = Query(
    #     ..., description="List of symptoms, e.g., symptoms=cough&symptoms=fever"
    # )
):
    """
    Query concepts matching one or more symptoms.
    Group results by semantic type (disorder, procedure, test, etc).
    """
    query = """
    MATCH (c:Concept)
    WHERE ANY(term IN $symptoms WHERE toLower(c.name) CONTAINS toLower(term))
    RETURN c.cui AS cui, c.name AS name, c.source AS source, labels(c) AS all_labels
    LIMIT 100
    """
    session = driver.session()
    result = session.run(query, symptoms=symptoms)
    concepts = result.data()
    session.close()
    print(f"Concepts: {concepts}")
    grouped: Dict[str, List[Dict]] = {}

    for concept in concepts:
        labels = remove_concept_label(concept["all_labels"])
        print(f"Labels: {labels}")
        primary_label = get_primary_label(labels, concept["name"])
        print(f"Primary Label: {primary_label}")
        symptoms = split_symptoms(concept["name"])
        print(f"symptoms: {symptoms}")
        # if len(symptoms) > 1:
        #     primary_label = "multiple"
        # elif len(symptoms) == 1:
        #     primary_label = symptoms[0]
        concept_entry = {
            "type": map_type(primary_label),
            "name": concept["name"],
            "source": concept["source"],
            "symptoms": symptoms,
        }

        if primary_label not in grouped:
            grouped[primary_label] = []
        grouped[primary_label].append(concept_entry)

    print(f"Grouped Concepts: {grouped}")
    return grouped


# @app.get("/summary", response_model=Dict[str, int], tags=["Concept Search"])
def get_summary(
    symptoms: List[str],
    # = Query(
    #     ..., description="List of symptoms, e.g., symptoms=cough&symptoms=fever"
    # )
):
    """
    Return a summary count of concepts matching one or more symptoms,
    grouped by semantic type (disorder, procedure, test, etc).
    """
    query = """
    MATCH (c:Concept)
    WHERE ANY(term IN $symptoms WHERE toLower(c.name) CONTAINS toLower(term))
    RETURN labels(c) AS all_labels, c.name AS name
    LIMIT 100
    """
    session = driver.session()
    result = session.run(query, symptoms=symptoms)
    concepts = result.data()
    session.close()
    print(f"Concepts: {concepts}")
    summary: Dict[str, int] = {}

    for concept in concepts:
        labels = remove_concept_label(concept["all_labels"])
        print(f"Labels: {labels}")
        primary_label = get_primary_label(labels, concept["name"])
        print(f"Primary Label: {primary_label}")

        if primary_label not in summary:
            summary[primary_label] = 0
        summary[primary_label] += 1

    print(f"Summary: {summary}")
    return summary


# # ---- MAIN ----
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


def build_patient_knowledge_graph(patient_profile):
    sex = patient_profile["sex"]
    age = patient_profile["age"]
    symptoms = patient_profile["symptoms"]
    history = patient_profile["history"]

    session = driver.session()

    # 1. Find diseases related to symptoms
    disease_symptom_query = """
    MATCH (s:Concept)-[:MANIFESTATION_OF|:ASSOCIATED_WITH]->(d:disorders)
    WHERE ANY(symptom IN $symptoms WHERE toLower(s.name) CONTAINS toLower(symptom))
    RETURN DISTINCT d.cui AS cui, d.name AS name
    """

    # 2. Find diseases related to history
    disease_history_query = """
    MATCH (h:Concept)-[:ASSOCIATED_WITH]->(d:disorders)
    WHERE ANY(history IN $history WHERE toLower(h.name) CONTAINS toLower(history))
    RETURN DISTINCT d.cui AS cui, d.name AS name
    """

    # Execute queries
    diseases_from_symptoms = session.run(
        disease_symptom_query, symptoms=symptoms
    ).data()
    diseases_from_history = session.run(disease_history_query, history=history).data()

    # Merge diseases (by CUI)
    disease_map = {}
    for d in diseases_from_symptoms:
        cui = d["cui"]
        if cui not in disease_map:
            disease_map[cui] = {
                "name": d["name"],
                "cui": cui,
                "linked_symptoms": [],
                "linked_history": [],
            }
        # Add symptoms linking
        disease_map[cui]["linked_symptoms"].append(d["name"])

    for d in diseases_from_history:
        cui = d["cui"]
        if cui not in disease_map:
            disease_map[cui] = {
                "name": d["name"],
                "cui": cui,
                "linked_symptoms": [],
                "linked_history": [],
            }
        # Add history linking
        disease_map[cui]["linked_history"].append(d["name"])

    # 3. Find tests and procedures linked to these diseases
    disease_cuis = list(disease_map.keys())
    test_proc_query = """
    MATCH (d:disorders)-[:HAS_METHOD|:MAY_TREAT|:MAY_PREVENT]->(p)
    WHERE d.cui IN $disease_cuis
    RETURN DISTINCT d.name AS disease, p.name AS procedure, labels(p) AS labels
    """

    tests = []
    procedures = []

    if disease_cuis:
        tests_procs = session.run(test_proc_query, disease_cuis=disease_cuis).data()
        for tp in tests_procs:
            labels = tp["labels"]
            if "procedures" in labels or "physiology" in labels:
                procedures.append(tp["procedure"])
            elif "tests" in labels or "activities_behaviors" in labels:
                tests.append(tp["procedure"])

    session.close()

    # ---- Build the Knowledge Graph JSON ----
    knowledge_graph = {
        "patient_profile": {
            "sex": sex,
            "age": age,
            "symptoms": symptoms,
            "history": history,
        },
        "possible_diseases": list(disease_map.values()),
        "suggested_tests": list(set(tests)),
        "suggested_procedures": list(set(procedures)),
    }

    return knowledge_graph

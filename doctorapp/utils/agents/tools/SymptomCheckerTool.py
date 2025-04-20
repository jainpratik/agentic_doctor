from langchain.tools import Tool
from neo4j import GraphDatabase


class SymptomToolNeo4j:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def __call__(self, input_text: str) -> str:
        symptoms = [s.strip().lower() for s in input_text.split(",")]
        query = """
        MATCH (d:Disease)-[:CAUSES]->(s:Symptom)
        WHERE toLower(s.name) IN $symptoms
        RETURN d.name AS disease, collect(s.name) AS matched_symptoms
        ORDER BY size(matched_symptoms) DESC
        LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query, symptoms=symptoms)
            formatted = []
            for row in result:
                disease = row["disease"]
                matched = ", ".join(row["matched_symptoms"])
                formatted.append(
                    f"Diagnosis: {disease}\n  → Supported symptoms: {matched}\n"
                )
            return "\n".join(formatted) or "No matching diseases found."

    def as_tool(self):
        return Tool(
            name="SymptomCheckerNeo4j",
            func=self,
            description="Use this tool to find diseases based on a comma-separated symptom list. Input format: 'fever, cough, fatigue'.",
        )

    def test(self):
        # Example test case
        input_text = "fever, cough"
        print(self(input_text))

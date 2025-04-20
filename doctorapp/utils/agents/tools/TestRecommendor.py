from langchain.tools import Tool
from neo4j import GraphDatabase


class TestRecommender:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def __call__(self, disease_name: str) -> str:
        query = """
        MATCH (d:Disease)-[:REQUIRES|:RECOMMENDED_TEST]->(p:Procedure)
        WHERE toLower(d.name) CONTAINS toLower($name)
        RETURN DISTINCT p.name AS procedure
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, name=disease_name.strip())
            procedures = [row["procedure"] for row in result]
            if not procedures:
                return "No specific procedures found for this disease."
            return f"🧪 Recommended tests for **{disease_name}**:\n" + "\n".join(
                f"- {p}" for p in procedures
            )

    def as_tool(self):
        return Tool(
            name="TestRecommender",
            func=self,
            description="Suggest tests or procedures for a disease. Input is the disease name, like 'pneumonia'.",
        )

    def test(self):
        # Example test case
        disease_name = "pneumonia"
        print(self(disease_name))

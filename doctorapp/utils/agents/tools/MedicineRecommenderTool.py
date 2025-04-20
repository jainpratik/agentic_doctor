from langchain.tools import Tool
from neo4j import GraphDatabase


class MedicationRecommender:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def __call__(self, disease: str) -> str:
        query = """
        MATCH (d:Disease)-[:TREATED_BY]->(m:Drug)
        WHERE toLower(d.name) CONTAINS toLower($name)
        RETURN DISTINCT m.name AS medication LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, name=disease.strip())
            meds = [row["medication"] for row in result]
            if not meds:
                return f"No known medications found for {disease}."
            return f"💊 Recommended medications for **{disease}**:\n" + "\n".join(
                f"- {m}" for m in meds
            )

    def as_tool(self):
        return Tool(
            name="MedicationRecommender",
            func=self,
            description="Use this tool to suggest medications for a disease. Input: disease name.",
        )

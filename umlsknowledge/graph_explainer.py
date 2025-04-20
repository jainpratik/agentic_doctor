from neo4j import GraphDatabase
import os

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))

def get_paths(cui):
    with driver.session() as session:
        result = session.run("""
            MATCH p=(a:Concept)-[r*1..2]->(b:Concept)
            WHERE a.CUI = $cui
            RETURN p LIMIT 5
        """, cui=cui)
        return [record["p"] for record in result]
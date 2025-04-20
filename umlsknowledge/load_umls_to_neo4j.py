# from neo4j import GraphDatabase
# import os
# import pandas as pd

# uri = os.getenv("NEO4J_URI")
# user = os.getenv("NEO4J_USER")
# password = os.getenv("NEO4J_PASSWORD")
# driver = GraphDatabase.driver(uri, auth=(user, password))

# concepts = pd.read_csv("data/umls_concepts.csv")

# def load_concepts(tx):
#     for _, row in concepts.iterrows():
#         tx.run(\"MERGE (c:Concept {CUI: $cui}) SET c.name = $name\", cui=row['CUI'], name=row['name'])

# with driver.session() as session:
#     session.write_transaction(load_concepts)


import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# Neo4j credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def run_query(query, params=None):
    with driver.session() as session:
        session.run(query, params or {})


def create_constraints():
    constraints = [
        ("Disease", "cui"),
        ("Symptom", "cui"),
        ("Drug", "drugbank_id"),
        ("Procedure", "cpt_code"),
        ("Test", "loinc_code"),
        ("Anatomy", "snomed_id"),
        ("Gene", "gene_id"),
        ("Pathway", "pathway_id"),
    ]
    for label, field in constraints:
        try:
            run_query(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{field} IS UNIQUE"
            )
        except Exception as e:
            print(f"Constraint for {label}.{field} failed: {e}")


def load_nodes(filepath, label, id_field, name_field="name", extra_fields=None):
    df = pd.read_csv(filepath)
    print(f"📄 Loaded {len(df)} rows from {filepath} for label {label}")
    extra_fields = extra_fields or []

    # query = f"""
    # MERGE (n:{label} {{{id_field}: ${id_field}}})
    # ON CREATE SET n.{name_field} = $name""" + "".join(
    #     [f", n.{field} = ${field}" for field in extra_fields]
    # )

    query = f"""
    MERGE (n:{label} {{{id_field}: ${id_field}}})
    ON CREATE SET n.{name_field} = $name""" + "".join(
        [f", n.{field} = ${field}" for field in extra_fields]
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {label}"):
        params = {id_field: row[id_field], "name": row[name_field]}
        for field in extra_fields:
            params[field] = row.get(field)
        print(f"⛓ Creating {label} node with {params}")
        print(
            f"⛓ Creating {label} node with {id_field}={row[id_field]}, name={row[name_field]}"
        )

        run_query(query, params)


def run_bulk_query(queries_with_params):
    with driver.session() as session:
        with session.begin_transaction() as tx:
            for query, params in queries_with_params:
                tx.run(query, params)


def load_relationships(filepath):
    df = pd.read_csv(filepath)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Relationships"):
        start_label = row["start_label"]
        start_id_field = row["start_id_field"]
        start_id = row["start_id"]

        end_label = row["end_label"]
        end_id_field = row["end_id_field"]
        end_id = row["end_id"]

        rel_type = row["type"].upper()

        query = f"""
        MATCH (a:{start_label} {{{start_id_field}: $start_id}})
        MATCH (b:{end_label} {{{end_id_field}: $end_id}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        print(
            f"🔗 Creating relationship {rel_type} from {start_label} ({start_id}) to {end_label} ({end_id})"
        )
        # Use parameterized query to prevent injection
        # and improve performance
        # Note: This is a simplified example; you may want to handle errors and retries
        # in a production setting.
        # Also, consider using a batch size for large datasets
        # to avoid overwhelming the database.
        # For example, you can use a batch size of 1000
        # and run the queries in chunks.
        # This is a basic implementation; you can improve it further.
        # by using a more sophisticated batching strategy.
        # For now, we will just run each query one by one
        # in a loop.
        run_query(query, {"start_id": start_id, "end_id": end_id})


def main():
    create_constraints()

    # Load nodes
    load_nodes("nodes/diseases.csv", "Disease", "cui")
    load_nodes("nodes/symptoms.csv", "Symptom", "cui")
    load_nodes("nodes/drugs.csv", "Drug", "drugbank_id")
    load_nodes("nodes/chemicals.csv", "Chemical", "drugbank_id")
    load_nodes("nodes/enzymes.csv", "Enzyme", "drugbank_id")
    load_nodes("nodes/procedures.csv", "Procedure", "cpt_code")
    load_nodes("nodes/tests.csv", "Test", "loinc_code")
    load_nodes("nodes/anatomy.csv", "Anatomy", "snomed_id")
    load_nodes("nodes/genes.csv", "Gene", "gene_id")
    # load_nodes("nodes/pathways.csv", "Pathway", "pathway_id", extra_fields=["source"])

    # Load relationships
    load_relationships("relationships.csv")

    print("✅ UMLS Graph Import Complete")


if __name__ == "__main__":
    main()
# Close the Neo4j driver connection
driver.close()

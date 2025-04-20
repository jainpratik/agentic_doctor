import os

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# === CONFIG ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
NODE_DIR = "nodes/"
REL_FILE = "relationships.csv"

# === CONNECT ===
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === NODE FILES MAP ===
node_files = {
    "diseases.csv": ("Disease", "CUI"),
    "symptoms.csv": ("Symptom", "CUI"),
    "drugs.csv": ("Drug", "CUI"),
    "chemicals.csv": ("Chemical", "CUI"),
    "enzymes.csv": ("Enzyme", "CUI"),
    "procedures.csv": ("Procedure", "CUI"),
    "tests.csv": ("Test", "CUI"),
    "anatomy.csv": ("Anatomy", "CUI"),
    "genes.csv": ("Gene", "CUI"),
}


def create_constraint(label, key):
    with driver.session() as session:
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        )


def load_nodes(label, id_field, filepath):
    df = pd.read_csv(filepath)
    if df.empty:
        print(f"⚪ Skipped empty file: {filepath}")
        return

    create_constraint(label, id_field)

    # Define allowed fields to avoid passing extra ones like SAUI, EXTRA, etc.
    allowed_fields = {
        id_field,
        "CUI",
        "STR",
        "name",
        "SAB",
        "TTY",
        "STY",
        "semantic_type",
    }

    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {label}"):
            # Create safe params dictionary
            params = {
                k: v for k, v in row.items() if k in allowed_fields and pd.notnull(v)
            }

            if id_field not in params:
                print(f"⚠️ Skipping row without {id_field}: {row}")
                continue

            # Generate Cypher SET clause dynamically
            set_clause = ", ".join(
                [f"n.{col} = ${col}" for col in params if col != id_field]
            )

            query = f"""
            MERGE (n:{label} {{{id_field}: ${id_field}}})
            ON CREATE SET {set_clause}
            """
            # print(f"⛓ Creating {label} node with {params}")
            session.run(query, params)


def load_relationships(rel_file):
    if not os.path.exists(rel_file):
        print("❌ relationships.csv not found.")
        return

    df = pd.read_csv(rel_file)
    if df.empty:
        print("⚪ Skipped empty relationships file.")
        return

    required_fields = {
        "start_label",
        "start_id_field",
        "start_id",
        "end_label",
        "end_id_field",
        "end_id",
        "type",
    }

    with driver.session() as session:
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="🔗 Creating relationships"
        ):
            # Skip rows missing critical fields
            if not required_fields.issubset(row.index) or row.isnull().any():
                print(f"⚠️ Skipping incomplete row: {row.to_dict()}")
                continue

            start_label = row["start_label"]
            start_id_field = row["start_id_field"]
            start_id = row["start_id"]
            end_label = row["end_label"]
            end_id_field = row["end_id_field"]
            end_id = row["end_id"]
            rel_type = row["type"]

            query = f"""
            MATCH (a:{start_label} {{{start_id_field}: $start_id}})
            MATCH (b:{end_label} {{{end_id_field}: $end_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            """

            session.run(query, {"start_id": start_id, "end_id": end_id})
            print(
                f"🔗 Created relationship {rel_type} from {start_label} ({start_id}) to {end_label} ({end_id})"
            )


def main():
    sample_data_path = "umls/sampledata/"
    real_data_path = "umls/realdata/"
    for file, (label, id_field) in node_files.items():
        path = os.path.join(real_data_path + NODE_DIR, file)
        print(f"🔍 Checking {label} file: {path}")
        if os.path.exists(path):
            print(f"🚀 Loading {label} from {file}")
            load_nodes(label, id_field, path)
        else:
            print(f"❌ File not found: {file}")

    # Load relationships
    # load_relationships(os.path.join(NODE_DIR, REL_FILE))

    print("✅ All nodes and relationships loaded into Neo4j.")


if __name__ == "__main__":
    main()

import pandas as pd
from tqdm import tqdm


def mrrel_to_csv(input_path: str, output_path: str) -> None:
    # Define column headers for MRREL.RRF
    columns = [
        "CUI1",
        "AUI1",
        "STYPE1",
        "REL",
        "CUI2",
        "AUI2",
        "STYPE2",
        "RELA",
        "RUI",
        "SRUI",
        "SAB",
        "SL",
        "RG",
        "DIR",
        "SUPPRESS",
        "CVF",
        "EXTRA",
    ]

    # Load MRREL.RRF
    df_rel = pd.read_csv(
        input_path + "MRREL.RRF",
        sep="|",
        names=columns,
        dtype=str,
        engine="python",
        encoding="latin-1",
    )

    # # Filter for meaningful medical relationships
    # valid_rels = ["CAUSES", "TREATS", "ASSOCIATED_WITH", "COMPLICATES", "CONTRAINDICATES"]
    # df_rel_filtered = df_rel[df_rel["REL"].isin(valid_rels)]
    # print(f"📄 Filtered {len(df_rel_filtered)} relationships")

    # Format for Neo4j loader
    # df_out = pd.DataFrame(
    #     {
    #         "start_label": "Concept",  # or use Disease/Symptom if known from MRSTY
    #         "start_id_field": "CUI",
    #         "start_id": df_rel_filtered["CUI1"],
    #         "type": df_rel_filtered["REL"],
    #         "end_label": "Concept",  # or specific like Symptom
    #         "end_id_field": "CUI",
    #         "end_id": df_rel_filtered["CUI2"],
    #     }
    # )

    # # Save relationships
    # df_out.to_csv("relationships.csv", index=False)
    # print(f"✅ Exported {len(df_out)} relationships")

    # Show the unique values of REL column to understand relationship types
    unique_rels = df_rel["REL"].dropna().unique()
    print(f"Unique REL types: {len(unique_rels)}")
    print(unique_rels)

    # Count occurrences of each REL type for inspection
    rel_counts = df_rel["REL"].value_counts().reset_index()
    rel_counts.columns = ["REL", "count"]

    rel_counts.head(20)

    # Define mapping for fallback relationship types
    rel_map = {
        "isa": "IS_A",
        "inverse_isa": "IS_A",
        "associated_with": "ASSOCIATED_WITH",
        "mapped_to": "MAPPED_TO",
        "has_component": "HAS_COMPONENT",
        "measures": "MEASURES",
        "has_permuted_term": "HAS_PERMUTED_TERM",
        "permuted_term_of": "PERMUTED_TERM_OF",
    }

    # Filter MRREL to only include mapped relationship types
    mrrel_filtered = df_rel[df_rel["REL"].isin(rel_map.keys())].copy()
    mrrel_filtered["REL_MAPPED"] = mrrel_filtered["REL"].map(rel_map)

    # Create relationships DataFrame for Neo4j loader
    relationships_csv = pd.DataFrame(
        {
            "start_label": "Concept",
            "start_id_field": "CUI",
            "start_id": mrrel_filtered["CUI1"],
            "type": mrrel_filtered["REL_MAPPED"],
            "end_label": "Concept",
            "end_id_field": "CUI",
            "end_id": mrrel_filtered["CUI2"],
        }
    )

    # Save to CSV
    output_path = output_path + "relationships.csv"
    relationships_csv.to_csv(output_path, index=False)
    print(f"✅ Exported {len(relationships_csv)} relationships to {output_path}")

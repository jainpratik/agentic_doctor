import json
import os
import re
import shutil
import zipfile
from random import sample

import pandas as pd
from mrconso_loader import mrconsoso_to_csv
from mrrel_loader import mrrel_to_csv
from mrsty_loader import mrsty_to_csv

sample_data_path = "umls/sampledata/"
real_data_path = "umls/realdata/"


# def create_relationships_df(input_path: str, output_path: str):

#     # Load relationships
#     df_rel = pd.read_csv(input_path + "relationships.csv")

#     # Load concepts
#     df_conso = pd.read_csv(input_path + "concepts.csv")

#     # Load semantic types
#     df_sty = pd.read_csv(input_path + "semantic_types.csv")

#     # Merge relationships with concepts and semantic types
#     df_rel_merged = df_rel.merge(df_conso, left_on="start_id", right_on="CUI")
#     df_rel_merged = df_rel_merged.merge(df_sty, left_on="CUI", right_on="CUI")

#     # Join on CUI
#     merged = df_conso.merge(df_sty[["CUI", "STY"]], on="CUI", how="left")

#     # Filter by semantic type
#     diseases = merged[merged["STY"] == "Disease or Syndrome"]
#     symptoms = merged[merged["STY"] == "Sign or Symptom"]
#     drugs = merged[merged["STY"] == "Pharmacologic Substance"]

#     # Export
#     diseases.to_csv(input_path + "nodes/diseases.csv", index=False)
#     symptoms.to_csv(input_path + "nodes/symptoms.csv", index=False)
#     drugs.to_csv(input_path + "nodes/drugs.csv", index=False)

#     # Merge on CUI to add semantic type to each concept
#     merged_df = df_conso.merge(df_sty[["CUI", "STY"]], on="CUI", how="left")

#     # Filter by known categories
#     drugs_df = merged_df[merged_df["STY"] == "Pharmacologic Substance"]
#     chemicals_df = merged_df[merged_df["STY"] == "Organic Chemical"]
#     enzymes_df = merged_df[merged_df["STY"] == "Enzyme"]

#     # Save CSVs to output
#     # Define output paths
#     diseases_path = output_path + "diseases.csv"
#     symptoms_path = output_path + "symptoms.csv"
#     drugs_path = output_path + "drugs.csv"
#     chemicals_path = output_path + "chemicals.csv"
#     enzymes_path = output_path + "enzymes.csv"

#     drugs_df.to_csv(drugs_path, index=False)
#     chemicals_df.to_csv(chemicals_path, index=False)
#     enzymes_df.to_csv(enzymes_path, index=False)

#     {
#         "drugs_csv": drugs_path,
#         "chemicals_csv": chemicals_path,
#         "enzymes_csv": enzymes_path,
#         "drugs_sample": drugs_df.head(2),
#     }

#     # Check if we have diseases or symptoms in the semantic types
#     diseases_df = merged_df[merged_df["STY"] == "Disease or Syndrome"]
#     symptoms_df = merged_df[merged_df["STY"] == "Sign or Symptom"]

#     # Save files if non-empty
#     if not diseases_df.empty:
#         diseases_df.to_csv(diseases_path, index=False)

#     if not symptoms_df.empty:
#         symptoms_df.to_csv(symptoms_path, index=False)

#     {
#         "diseases_count": len(diseases_df),
#         "symptoms_count": len(symptoms_df),
#         "diseases_path": diseases_path if not diseases_df.empty else "No data found",
#         "symptoms_path": symptoms_path if not symptoms_df.empty else "No data found",
#     }


def create_relationships_df(input_path: str, output_path: str):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load files
    # df_rel = pd.read_csv(os.path.join(input_path, "relationships.csv"))
    df_conso = pd.read_csv(os.path.join(input_path, "concepts.csv"))
    df_sty = pd.read_csv(os.path.join(input_path, "semantic_types.csv"))

    # Merge concepts with semantic types
    merged_df = df_conso.merge(df_sty[["CUI", "STY"]], on="CUI", how="left")

    # Define semantic type → output file mapping
    semantic_type_map = {
        "Disease or Syndrome": "diseases.csv",
        "Sign or Symptom": "symptoms.csv",
        "Pharmacologic Substance": "drugs.csv",
        "Organic Chemical": "chemicals.csv",
        "Enzyme": "enzymes.csv",
        "Therapeutic or Preventive Procedure": "procedures.csv",
        "Laboratory Procedure": "tests.csv",
        "Diagnostic Procedure": "tests.csv",
        "Body Part, Organ, or Organ Component": "anatomy.csv",
        "Gene or Genome": "genes.csv",
    }

    # Group and export
    exported_files = {}
    for sty, filename in semantic_type_map.items():
        filtered = merged_df[merged_df["STY"] == sty]
        if not filtered.empty:
            filepath = os.path.join(output_path, filename)
            filtered.to_csv(filepath, index=False)
            exported_files[sty] = filepath

    # Optional: print summary
    print(f"✅ Exported node files:")
    for sty, path in exported_files.items():
        print(f"- {sty}: {path} ({len(pd.read_csv(path))} rows)")

    return exported_files


# Load MRCONSO.RRF
# mrconsoso_to_csv(input_path=sample_data_path, output_path=sample_data_path)
# mrrel_to_csv(input_path=sample_data_path, output_path=sample_data_path)
# mrsty_to_csv(input_path=sample_data_path, output_path=sample_data_path)
# create_relationships_df(
#     input_path=sample_data_path,
#     output_path=sample_data_path,
# )


# mrconsoso_to_csv(input_path=real_data_path, output_path=real_data_path)
mrrel_to_csv(input_path=real_data_path, output_path=real_data_path)
# mrsty_to_csv(input_path=real_data_path, output_path=real_data_path)
# create_relationships_df(
#     input_path=real_data_path,
#     output_path=real_data_path + "nodes/",
# )

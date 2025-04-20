import pandas as pd


def mrconsoso_to_csv(input_path: str, output_path: str) -> pd.DataFrame:
    columns = [
        "CUI",
        "LAT",
        "TS",
        "LUI",
        "STT",
        "SUI",
        "ISPREF",
        "AUI",
        "SAUI",
        "SCUI",
        "SDUI",
        "SAB",
        "TTY",
        "CODE",
        "STR",
        "SRL",
        "SUPPRESS",
        "CVF",
        "EXTRA",  # Placeholder for trailing pipe
    ]

    df_conso = pd.read_csv(
        input_path + "MRCONSO.RRF",
        sep="|",
        names=columns,
        dtype=str,
        engine="python",
        encoding="latin-1",  # 🔥 FIX: Handle special characters
        on_bad_lines="skip",  # Skip malformed lines (pandas >=1.3)
        quoting=3,  # Ignore any quote issues
        skip_blank_lines=True,  # Skip blank lines
        skiprows=1,  # Skip header row
    )

    # Filter English only
    df_conso = df_conso[df_conso["LAT"] == "ENG"]

    # Save cleaned file
    df_conso.to_csv(output_path + "concepts.csv", index=False)
    print(f"✅ Exported {len(df_conso)} concepts to concepts.csv")

    return df_conso

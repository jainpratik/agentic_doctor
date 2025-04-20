import pandas as pd
from tqdm import tqdm


def mrsty_to_csv(input_path: str, output_path: str) -> None:
    """
    Convert the MRSTY file to a CSV file.

    Args:
        input_path (str): Path to the MRSTY file.
        output_path (str): Path to the output CSV file.
    """

    # Define correct column names for MRSTY.RRF including a placeholder for trailing pipe
    sty_columns = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "EXTRA"]

    # Load and parse MRSTY.RRF correctly
    df_sty_fixed = pd.read_csv(
        input_path + "MRSTY.RRF",
        sep="|",
        names=sty_columns,
        engine="python",
        encoding="latin-1",
    )

    # Display unique semantic types with counts
    sty_counts = df_sty_fixed["STY"].value_counts().reset_index()
    sty_counts.columns = ["STY", "count"]

    sty_counts.head(20)
    output_path = output_path + "semantic_types.csv"
    df_sty_fixed.to_csv(output_path, index=False)
    print(f"✅ Exported {len(df_sty_fixed)} relationships to {output_path}")

from pathlib import Path

import pandas as pd


# Configuration
INPUT_FILE = Path(
    "../data/raw/PepBenchData-50_antimicrobial_E_coli_mic_all.csv"
)

OUTPUT_FILE = Path(
    "../data/benchmark/mic.csv"
)

MAX_LENGTH = 96

STANDARD_AA = {
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V"
}


# Validation
def contains_only_standard_aa(sequence: str) -> bool:
    """
    Check whether a peptide contains only
    canonical amino acids.
    """
    return all(
        aa in STANDARD_AA
        for aa in sequence
    )


def filter_noncanonical(df: pd.DataFrame) -> pd.DataFrame:

    before = len(df)

    mask = df["sequence"].apply(
        contains_only_standard_aa
    )

    filtered_df = df[mask].copy()

    removed = before - len(filtered_df)

    print(
        f"Canonical AA filtering: "
        f"{before} -> {len(filtered_df)} "
        f"(removed {removed}, "
        f"{removed / before:.2%})"
    )

    return filtered_df


def filter_by_length(
    df: pd.DataFrame,
    max_length: int = 96
) -> pd.DataFrame:

    before = len(df)

    mask = (
        df["sequence"]
        .str.len()
        <= max_length
    )

    filtered_df = df[mask].copy()

    removed = before - len(filtered_df)

    print(
        f"Length filtering <= {max_length}: "
        f"{before} -> {len(filtered_df)} "
        f"(removed {removed}, "
        f"{removed / before:.2%})"
    )

    return filtered_df


def check_alphabet(df: pd.DataFrame):

    alphabet = set(
        "".join(df["sequence"])
    )

    print(
        f"\nUnique residues ({len(alphabet)}):"
    )
    print(sorted(alphabet))

    unexpected = alphabet - STANDARD_AA

    if unexpected:
        print(
            f"\nUnexpected residues found:"
        )
        print(sorted(unexpected))
    else:
        print(
            "\nOnly canonical amino acids remain."
        )


def main():

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(
        f"Initial dataset size: {len(df)}"
    )

    if "sequence" not in df.columns:
        raise KeyError(
            f"Column 'sequence' not found.\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    lengths = df["sequence"].str.len()

    print("\nInitial length statistics:")

    print(
        f"min={lengths.min()}, "
        f"mean={lengths.mean():.1f}, "
        f"median={lengths.median():.1f}, "
        f"max={lengths.max()}"
    )

    # Canonical amino acid filtering
    df = filter_noncanonical(df)

    # Length filtering
    df = filter_by_length(
        df,
        max_length=MAX_LENGTH
    )

    check_alphabet(df)

    # Rename sequence column
    df = df.rename(
        columns={
            "sequence": "seq"
        }
    )

    lengths = df["seq"].str.len()

    print("\nFinal length statistics:")

    print(
        f"min={lengths.min()}, "
        f"mean={lengths.mean():.1f}, "
        f"median={lengths.median():.1f}, "
        f"max={lengths.max()}"
    )

    print(
        f"\nFinal dataset size: {len(df)}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\nExample:")
    print(df.head())


if __name__ == "__main__":
    main()
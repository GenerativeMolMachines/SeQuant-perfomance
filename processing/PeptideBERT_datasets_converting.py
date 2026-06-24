import numpy as np
import pandas as pd
from pathlib import Path


# PeptideBERT vocabulary
TOKENS = [
    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
    'L', 'A', 'G', 'V', 'E', 'S', 'I', 'K', 'R',
    'D', 'T', 'P', 'N', 'Q', 'F', 'Y', 'M', 'H',
    'C', 'W', 'X', 'U', 'B', 'Z', 'O'
]

# Keep only valid amino acid tokens
IDX2AA = {
    idx: token
    for idx, token in enumerate(TOKENS)
    if len(token) == 1
}


# Decoding funcs
def decode_sequence(encoded_sequence):
    """
    Convert integer-encoded peptide into amino acid sequence.
    Parameters
    encoded_sequence : np.ndarray
        Array of token ids.
    Returns
    str
        Amino acid sequence.
    """
    sequence = []

    for token in encoded_sequence:
        # Ignore padding
        if token == 0:
            continue

        if token not in IDX2AA:
            raise ValueError(
                f"Unknown token {token}. "
                f"Available tokens: {sorted(IDX2AA.keys())}"
            )
        sequence.append(IDX2AA[token])

    return ''.join(sequence)


def load_npz_sequences(npz_path):
    """
    Load and decode sequences from npz file.
    """
    data = np.load(npz_path)

    if "arr_0" not in data:
        raise KeyError(
            f"Expected key 'arr_0' in {npz_path}. "
            f"Found: {data.files}"
        )
    encoded_sequences = data["arr_0"]
    return [
        decode_sequence(seq)
        for seq in encoded_sequences
    ]


def filter_by_length(df, max_length=96):
    """
    Remove sequences longer than max_length.
    Parameters
    df : pd.DataFrame
        DataFrame with columns ['seq', 'label'].
    max_length : int
        Maximum allowed sequence length.
    Returns
    pd.DataFrame
    """
    original_size = len(df)
    pos_before = (df.label == 1).sum()
    neg_before = (df.label == 0).sum()

    filtered_df = df[
        df["seq"].str.len() <= max_length
    ].copy()

    pos_after = (filtered_df.label == 1).sum()
    neg_after = (filtered_df.label == 0).sum()
    removed = original_size - len(filtered_df)

    print(
        f"Length filtering (<= {max_length} aa): "
        f"removed {removed} samples "
        f"({removed / original_size:.2%})"
    )
    print(
        f"Positive: {pos_before} -> {pos_after} "
        f"(removed {pos_before - pos_after})"
    )

    print(
        f"Negative: {neg_before} -> {neg_after} "
        f"(removed {neg_before - neg_after})"
    )

    return filtered_df


# Dataset creation
def build_dataset(task_name,
                  input_dir="../data/raw",
                  output_dir="../data/benchmark",
                  shuffle=True,
                  random_state=42):
    """
    Build CSV dataset from positive and negative npz files.
    Expected files:
        {task_name}-positive.npz
        {task_name}-negative.npz

    Output:
        {task_name}.csv

    Columns:
        seq,label
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    pos_file = input_dir / f"{task_name}-positive.npz"
    neg_file = input_dir / f"{task_name}-negative.npz"

    print(f"\nProcessing {task_name}")

    positive_sequences = load_npz_sequences(pos_file)
    negative_sequences = load_npz_sequences(neg_file)

    df_positive = pd.DataFrame({
        "seq": positive_sequences,
        "label": 1
    })

    df_negative = pd.DataFrame({
        "seq": negative_sequences,
        "label": 0
    })

    df = pd.concat(
        [df_positive, df_negative],
        ignore_index=True
    )

    lengths = df["seq"].str.len()

    print(
        f"Length statistics:\n"
        f"min={lengths.min()}, "
        f"mean={lengths.mean():.1f}, "
        f"median={lengths.median():.1f}, "
        f"max={lengths.max()}"
    )

    df = filter_by_length(
        df,
        max_length=96
    )

    if shuffle:
        df = df.sample(
            frac=1,
            random_state=random_state
        ).reset_index(drop=True)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / f"{task_name}.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Positive samples before filtration: {len(df_positive)}")
    print(f"Negative samples before filtration: {len(df_negative)}")
    print(f"Total samples    : {len(df)}")
    print(f"Saved to         : {output_file}")

    print("\nExample:")
    print(df.iloc[0])

    return df


def main():

    tasks = [
        "hemo",
        "sol",
        "nf"
    ]

    for task in tasks:
        build_dataset(
            task_name=task,
            input_dir="../data/raw",
            output_dir="../data/benchmark"
        )


if __name__ == "__main__":
    main()
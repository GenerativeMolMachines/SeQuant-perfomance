import os
import sys
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from tqdm import tqdm
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from Bio.Align import substitution_matrices
from p2smi.utilities import smilesgen

from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    AutoTokenizer,
    TFAutoModel,
    AutoModelForMaskedLM,
    AutoModel
)

import esm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

import ankh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/dcBiLSTM")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../models/PeptideCLM")))

from my_tokenizers import SMILES_SPE_Tokenizer
from preset_tools import data_processing
from dcBiLSTM import *


# Config
DATASETS_CLASSIFICATION = [
    "antimic",
    "antidia",
    "antiinf",
    "antiox",
    "hemo",
    "nf",
    "sol",
]
DATASETS_REGRESSION = [
    "mic"
]

INPUT_DIR = "../data/benchmark"
OUTPUT_DIR = "../data/encoded"

MAX_LEN = 96
ENABLE_SEQUANT_API = False

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
monomer_dict = {
    'A': 'CC(N)C(=O)O', 'R': 'NC(N)=NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O', 'C': 'NC(CS)C(=O)O', 'Q': 'NC(=O)CCC(N)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1cnc[nH]1)C(=O)O',
    'I': 'CCC(C)C(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
    'S': 'NC(CO)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O', 'O': 'CC1CC=NC1C(=O)NCCCCC(N)C(=O)O',
    'U': 'NC(C[Se])C(=O)O'
}

tqdm.pandas()

# FUNCTIONS

# Baselines
def random_embeddings(df, emb_dim=92, seed=42):
    rng = np.random.default_rng(seed)
    random_emb = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(len(df), emb_dim)
    )

    return pd.concat(
        [df[["seq", "label"]], pd.DataFrame(random_emb)], axis=1
    )


def scrambled_embeddings(df, emb_start_col=2, seed=42):
    rng = np.random.default_rng(seed)
    emb = df.iloc[:, emb_start_col:].to_numpy()
    shuffled_idx = rng.permutation(len(df))
    emb_scrambled = emb[shuffled_idx]

    return pd.concat(
        [
            df.iloc[:, :emb_start_col].reset_index(drop=True),
            pd.DataFrame(emb_scrambled, columns=df.columns[emb_start_col:])
        ],
        axis=1
    )


# Conventional encoding strategies
def one_hot_encode(seq):
    encoder = OneHotEncoder(
        categories=[list(amino_acids)],
        sparse_output=False,
        dtype=int
    )
    arr = np.array(list(seq)).reshape(-1, 1)
    return encoder.fit_transform(arr).flatten()


def threemers_encode(seq):
    k = 3
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    kmer_space = [''.join(p) for p in product(amino_acids, repeat=k)]
    kmer_to_idx = {k: i for i, k in enumerate(kmer_space)}
    return [kmer_to_idx[k] for k in kmers]


def blosum62_encode(seq):
    blosum62 = substitution_matrices.load("BLOSUM62")
    vec = []

    for i in range(len(seq)-1):
        pair = (seq[i], seq[i+1])
        if pair in blosum62:
            vec.append(blosum62[pair])
        elif (pair[1], pair[0]) in blosum62:
            vec.append(blosum62[(pair[1], pair[0])])
        else:
            vec.append(0)

    return vec


# Convert seq to smiles
def peptide_to_smiles(seq):
    result = smilesgen.constrained_peptide_smiles(seq, "")
    return result[2]


# pLMs
def prot_t5_encode(seqs, batch_size=16):
    seqs = [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in seqs]
    out = []

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size)):
            batch = seqs[i:i+batch_size]

            enc = prott5_tokenizer(
                batch,
                return_tensors="pt",
                padding=True
            )

            ids = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)

            h = prott5_model(input_ids=ids, attention_mask=mask).last_hidden_state

            mask = mask.unsqueeze(-1)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1)

            out.append(emb.cpu().numpy())

    return np.vstack(out)


def protbert_encode(seqs, batch_size=16):
    seqs = [" ".join(list(s)) for s in seqs]
    out = []

    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i+batch_size]

        enc = protbbert_tokenizer(
            batch,
            return_tensors="tf",
            padding=True
        )

        ids = enc["input_ids"]
        mask = enc["attention_mask"]

        h = protbbert_model(ids, attention_mask=mask).last_hidden_state

        mask = tf.cast(mask[:, :, None], tf.float32)

        h = h * mask
        emb = tf.reduce_sum(h, axis=1) / tf.reduce_sum(mask, axis=1)

        out.append(emb.numpy())

    return np.vstack(out)


def esmc_encode(seqs):
    out = []

    with torch.no_grad():
        for seq in tqdm(seqs):

            p = ESMProtein(sequence=seq)
            t = esmc_model.encode(p)

            logits = esmc_model.logits(
                t,
                LogitsConfig(
                    sequence=True,
                    return_embeddings=True
                )
            )

            token_emb = logits.embeddings.squeeze(0)
            emb = token_emb.mean(dim=0)
            out.append(emb.cpu().numpy())

    return np.stack(out)


def ankh_encode(seqs, batch_size=16):
    seqs = [
        " ".join(list(re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", s)))
        for s in seqs
    ]

    out = []

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size)):
            batch = seqs[i:i+batch_size]

            enc = ankh_tokenizer(
                batch,
                return_tensors="pt",
                padding=True
            )

            ids = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)

            h = ankh_model(
                input_ids=ids,
                attention_mask=mask
            ).last_hidden_state

            mask = mask.unsqueeze(-1)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1)

            out.append(emb.cpu().numpy())

    return np.vstack(out)


def pepmlm_encode(seqs, batch_size=16):
    out = []

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size)):
            batch = seqs[i:i + batch_size]

            enc = pepmlm_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            ids = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)

            outputs = pepmlm_model(
                input_ids=ids,
                attention_mask=mask
            )

            h = outputs.hidden_states[-1]
            mask = mask.unsqueeze(-1)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1)
            out.append(
                emb.cpu().numpy()
            )
    return np.vstack(out)


def peptideclm_encode(seqs, batch_size=16):

    smiles_list = [
        peptide_to_smiles(seq)
        for seq in seqs
    ]

    out = []

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size)):
            batch = smiles_list[i:i+batch_size]

            enc = peptideclm_tokenizer(
                batch,
                return_tensors="pt",
                padding=True
            )

            ids = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)

            outputs = peptideclm_model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True
            )

            h = outputs.hidden_states[-1]
            mask = mask.unsqueeze(-1)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1)
            out.append(
                emb.cpu().numpy()
            )

    return np.vstack(out)


# Encodings
def encode_plm(df, enc_fn):
    emb = enc_fn(df["seq"].tolist())
    return pd.concat(
        [df, pd.DataFrame(emb, index=df.index)],
        axis=1
    )


def encode_classical(df, enc_fn, pad_value=0):
    encoded_data = df['seq'].progress_apply(enc_fn)
    max_len = max(encoded_data.apply(len))
    encoded_data = encoded_data.apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant', constant_values=pad_value))
    encoded_df = pd.DataFrame(encoded_data.tolist(), index=df.index)

    return pd.concat([df, encoded_df], axis=1)


# SeQuant models
def prepare_bilstm_input(seqs):
    batch_data = [
        (seq, 0)
        for seq in seqs
    ]

    x, _ = data_processing(
        batch_data=batch_data,
        monomer_dict=monomer_dict,
        max_len=MAX_LEN
    )

    return x


def bilstm_encode(df, encoder):
    seqs = df["seq"].tolist()
    x = prepare_bilstm_input(seqs)
    emb = encoder.predict(x, verbose=0, batch_size=32)

    return pd.concat([df, pd.DataFrame(emb, index=df.index)], axis=1)


def load_bilstm(path):
    model = tf.keras.models.load_model(path)
    return model.get_layer("model")


def load_cbilstm(path):
    model = tf.keras.models.load_model(path)
    return model.get_layer("model")


def load_dcbilstm(path):
    model = tf.keras.models.load_model(path)
    return model.encoder


# Aggregation
def run_dataset(name, df, enc_name, enc_fn):

    print(f"\n[{name}] -> {enc_name}")

    if enc_name in pLMs:
        df_out = encode_plm(df, enc_fn)
    else:
        df_out = encode_classical(df, enc_fn)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(
        f"{OUTPUT_DIR}/{name}_{enc_name}.csv",
        index=False
    )


def run_bilstm(name, df, model, model_name):

    print(f"\n[{name}] -> {model_name}")

    df_out = bilstm_encode(df, model)
    df_out.to_csv(
        f"../data/embeddings_SeQuant/{name}_{model_name}.csv",
        index=False
    )


ENCODERS = {
    "one_hot": one_hot_encode,
    "threemers": threemers_encode,
    "blosum62": blosum62_encode,
    "protbert": protbert_encode,
    "prott5": prot_t5_encode,
    "esmc": esmc_encode,
    "ankh_large": ankh_encode,
    "pepmlm": pepmlm_encode,
    "peptideclm": peptideclm_encode
}
pLMs = ["protbert", "prott5", "esmc", "ankh_large", "pepmlm", "peptideclm"]


# load pLMs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nLoading model ProtT5...")
# ProtT5
PROTT5_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
prott5_tokenizer = T5Tokenizer.from_pretrained(PROTT5_NAME)
prott5_model = T5EncoderModel.from_pretrained(PROTT5_NAME).to(DEVICE)
prott5_model.eval()
if DEVICE.type == "cpu":
    prott5_model = prott5_model.float()

print("\nLoading model ProtBERT...")
# ProtBERT
PROTBERT_NAME = "Rostlab/prot_bert"
protbbert_tokenizer = AutoTokenizer.from_pretrained(PROTBERT_NAME)
protbbert_model = TFAutoModel.from_pretrained(PROTBERT_NAME, from_pt=True)

print("\nLoading model ESM-C...")
# ESM-C
esmc_model = ESMC.from_pretrained("esmc_600m").to(DEVICE)
esmc_model.eval()

print("\nLoading model Ankh...")
# Ankh
ankh_model, ankh_tokenizer = ankh.load_large_model()
ankh_model = ankh_model.to(DEVICE)
ankh_model.eval()

print("\nLoading model PepMLM...")
# PepMLM
PEPMLM_NAME = "ChatterjeeLab/PepMLM-650M"
pepmlm_tokenizer = AutoTokenizer.from_pretrained(
    PEPMLM_NAME
)
pepmlm_model = AutoModelForMaskedLM.from_pretrained(
    PEPMLM_NAME,
    output_hidden_states=True
).to(DEVICE)
pepmlm_model.eval()

print("\nLoading model PeptideCLM")
# PeptideCLM
PEPTIDECLM_NAME = "aaronfeller/PeptideCLM-23M-all"
peptideclm_tokenizer = SMILES_SPE_Tokenizer(
    "../models/PeptideCLM/new_vocab.txt",
    "../models/PeptideCLM/new_splits.txt"
)
peptideclm_model = AutoModel.from_pretrained(
    PEPTIDECLM_NAME
).to(DEVICE)
peptideclm_model.eval()


def main():

    datasets = DATASETS_CLASSIFICATION + DATASETS_REGRESSION

    print("\nLoading BiLSTM...")
    bilstm = load_bilstm(
        "../models/biLSTM/checkpoint_biLSTM"
    )

    print("\nLoading cBiLSTM...")
    cbilstm = load_cbilstm(
        "../models/cBiLSTM/checkpoint_biLSTM_contrastive_16"
    )

    print("\nLoading dcBiLSTM...")
    dcbilstm = load_dcbilstm(
        "../models/dcBiLSTM/checkpoint_biLSTM_contrastive_diag_loss_16_16_1_mini"
    )

    for name in datasets:
        df = pd.read_csv(f"{INPUT_DIR}/{name}.csv")

        # classical + PLM
        for enc_name, enc_fn in ENCODERS.items():
            run_dataset(name, df, enc_name, enc_fn)

        # biLSTM family
        run_bilstm(name, df, bilstm, "bilstm")
        run_bilstm(name, df, cbilstm, "cbilstm")
        run_bilstm(name, df, dcbilstm, "dcbilstm")

        # SeQuant (optional)
        if ENABLE_SEQUANT_API:
            print("SeQuant API disabled in this version")

        df_sequant = bilstm_encode(df, dcbilstm)
        random_df = random_embeddings(df_sequant, emb_dim=92)
        scrambled_df = scrambled_embeddings(df_sequant)

        random_df.to_csv(
            f"../data/encoded/{name}_random92.csv",
            index=False
        )
        scrambled_df.to_csv(
            f"../data/encoded/{name}_scrambled92.csv",
            index=False
        )

    print("\nDone")

if __name__ == "__main__":
    main()
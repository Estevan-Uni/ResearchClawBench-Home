"""Shared paths, palette and loaders for the Uncalled4 reanalysis."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report", "images")
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

SEED = 20241125
BASES = "ACGT"
BASE_COLORS = {"A": "#2a9d8f", "C": "#4361ee", "G": "#e9a03b", "T": "#d62839"}
TOOL_COLORS = {"Uncalled4": "#1f5aa6", "f5c": "#e07b39", "Nanopolish": "#7a7a7a", "Tombo": "#9c6ade"}

MODELS = {
    "DNA r9.4.1 (6-mer)": "dna_r9.4.1_400bps_6mer_uncalled4.csv",
    "DNA r10.4.1 (9-mer)": "dna_r10.4.1_400bps_9mer_uncalled4.csv",
    "RNA001 (5-mer)": "rna_r9.4.1_70bps_5mer_uncalled4.csv",
    "RNA004 (9-mer)": "rna004_130bps_9mer_uncalled4.csv",
}
SHORT = {"DNA r9.4.1 (6-mer)": "dna_r941", "DNA r10.4.1 (9-mer)": "dna_r1041",
         "RNA001 (5-mer)": "rna001", "RNA004 (9-mer)": "rna004"}


def load_model(name):
    """Return (df, K, X) where df is indexed by explicit base-4 code
    (position 0 = most significant digit) and X is an (N, K) int array of base indices."""
    df = pd.read_csv(os.path.join(DATA, MODELS[name]))
    df["kmer"] = df["kmer"].str.upper().str.replace("U", "T")
    K = len(df["kmer"].iloc[0])
    X = np.array([[BASES.index(c) for c in s] for s in df["kmer"].values], dtype=np.int8) \
        if K <= 6 else np.frombuffer("".join(df["kmer"].values).encode(), dtype=np.uint8).reshape(-1, K)
    if K > 6:
        lut = np.zeros(256, dtype=np.int8)
        for i, b in enumerate(BASES):
            lut[ord(b)] = i
        X = lut[X]
    code = np.zeros(len(df), dtype=np.int64)
    for p in range(K):
        code = code * 4 + X[:, p]
    df["code"] = code
    order = np.argsort(code)
    df = df.iloc[order].reset_index(drop=True)
    X = X[order]
    assert np.array_equal(df["code"].values, np.arange(4 ** K)), "incomplete k-mer table"
    return df, K, X


def code_of(kmer):
    c = 0
    for ch in kmer:
        c = c * 4 + BASES.index(ch)
    return c

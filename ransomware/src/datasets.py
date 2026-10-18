"""
Loaders for the three ransomware datasets, returning one common structure.

Every dataset becomes a binary task - ransomware (1) vs benign (0) - with a fixed, sealed
train/test split and, for each feature, the analysis TIER it needs:

    static   - read from the file without running it (PE header, extracted strings)
    dynamic  - only observable by executing the sample in a sandbox
    network  - observed passively from network traffic (UGRansome)

The tier drives objective f1 (detection time) - see cost.py.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RANDOM_STATE = 42


@dataclass
class Dataset:
    name: str
    title: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    feature_tier: dict[str, str]
    notes: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# --------------------------------------------------------------------------
# MLRan - behavioural dataset, authors' RFE-selected 483 binary features
# --------------------------------------------------------------------------

def load_mlran() -> Dataset:
    """MLRan (Onwuegbuche et al., JNCA 2026). Uses the authors' own train/test split,
    because their RFE feature selection was fitted on their training data - re-splitting
    would leak test information into the feature set."""
    d = DATA / "mlran"
    tr = pd.read_csv(d / "MLRan_X_train_RFE.csv")
    te = pd.read_csv(d / "MLRan_X_test_RFE.csv")
    names = json.loads((d / "RFE_selected_feature_names_dic.json").read_text())

    label_cols = ["sample_id", "sample_type", "family_label", "type_label"]
    ids = [c for c in tr.columns if c not in label_cols]
    feature_names = [names[c] for c in ids]
    # STRING features are printable strings extracted from the file itself - static.
    # Every other group (API, REG, FILE, DIRECTORY, SYSTEM, DROP, SIGNATURE) comes from
    # running the sample in the Cuckoo sandbox - dynamic.
    tier = {n: ("static" if n.startswith("STRING:") else "dynamic") for n in feature_names}

    return Dataset(
        name="mlran", title="MLRan",
        X_train=tr[ids].to_numpy(np.float32), y_train=tr["sample_type"].to_numpy(np.int8),
        X_test=te[ids].to_numpy(np.float32), y_test=te["sample_type"].to_numpy(np.int8),
        feature_names=feature_names, feature_tier=tier,
        notes={"split": "authors' predefined split"},
    )


# --------------------------------------------------------------------------
# RD2024 - Ransomware Dataset 2024 (Hussain, Zenodo 13890887)
# --------------------------------------------------------------------------

DYNAMIC_COLS_RD2024 = [
    "registry_read", "registry_write", "registry_delete", "registry_total",
    "network_threats", "network_dns", "network_http", "network_connections",
    "processes_malicious", "processes_suspicious", "processes_monitored", "total_procsses",
    "files_malicious", "files_suspicious", "files_text", "files_unknown",
    "dlls_calls", "apis",
]
CATEGORICAL_RD2024 = ["PEType", "MachineType", "Magic", "Subsystem"]
LIST_COLS_RD2024 = ["DllCharacteristics", "text_Characteristics", "rdata_Characteristics"]
# Sandbox VERDICT counts - the sandbox's own malicious/suspicious judgements, not raw
# behaviour. processes_malicious alone separates the classes at AUC 0.996 in the audit:
# keeping these would make our detector re-read another detector's answer.
VERDICT_COLS_RD2024 = ["processes_malicious", "processes_suspicious",
                       "files_malicious", "files_suspicious", "network_threats"]
DROP_RD2024 = ["md5", "sha1", "Class", "Category", "Family",
               "file_extension", "magic_number"] + VERDICT_COLS_RD2024


def _hex(v):
    if pd.isna(v):
        return np.nan
    s = str(v).split(" ")[0].strip()
    try:
        return float(int(s, 16)) if s.lower().startswith("0x") else float(s)
    except ValueError:
        return np.nan


def _parse_list(v):
    try:
        out = ast.literal_eval(v)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def load_rd2024() -> Dataset:
    df = pd.read_csv(DATA / "rd2024" / "Final_Dataset_without_duplicate.csv",
                     low_memory=False, dtype=str)
    df = df[df["Category"].isin(["Ransomware", "Benign"])].reset_index(drop=True)
    y = (df["Category"] == "Ransomware").astype(np.int8).to_numpy()

    cols: dict[str, np.ndarray] = {}
    tier: dict[str, str] = {}
    for c in df.columns:
        if c in DROP_RD2024:
            continue
        if c in DYNAMIC_COLS_RD2024:
            cols[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            tier[c] = "dynamic"
        elif c in CATEGORICAL_RD2024:
            for val in sorted(df[c].dropna().unique()):
                name = f"{c}={val}"
                cols[name] = (df[c] == val).astype(float).to_numpy()
                tier[name] = "static"
        elif c in LIST_COLS_RD2024:
            parsed = df[c].map(_parse_list)
            flags = sorted({f for lst in parsed for f in lst})
            for f in flags:
                name = f"{c}:{f}"
                cols[name] = parsed.map(lambda lst, f=f: float(f in lst)).to_numpy()
                tier[name] = "static"
        else:
            cols[c] = df[c].map(_hex).to_numpy(float)
            tier[c] = "static"

    X = pd.DataFrame(cols).fillna(-1).to_numpy(np.float32)
    names = list(cols)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return Dataset(
        name="rd2024", title="Ransomware Dataset 2024",
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        feature_names=names, feature_tier=tier,
        notes={"split": "stratified 80/20, random_state=42",
               "rows_used": int(len(df)), "families": int(df.loc[y == 1, "Family"].nunique()),
               "excluded_verdict_features": VERDICT_COLS_RD2024},
    )


# --------------------------------------------------------------------------
# UGRansome - network-traffic dataset (Nkongolo et al., 2021) - Kaggle download
# --------------------------------------------------------------------------

def load_ugransome() -> Dataset:
    from ugransome import build_ugransome   # defined once the CSV is available
    return build_ugransome(DATA / "ugransome")


LOADERS = {"mlran": load_mlran, "rd2024": load_rd2024, "ugransome": load_ugransome}


def load(name: str) -> Dataset:
    return LOADERS[name]()

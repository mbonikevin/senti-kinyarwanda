"""load and clean the afrisenti kinyarwanda tweets"""

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
GITHUB_BASE = ("https://raw.githubusercontent.com/afrisenti-semeval/"
               "afrisent-semeval-2023/main/data/kin")
SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ID_TO_LABEL = {0: "positive", 1: "neutral", 2: "negative"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
LABEL_ORDER = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff️‍]+")

def clean_text(text, strip_emoji=False, fix_elongation=False):
    """cleans one tweet the two flags stay off unless e6 turns them on"""
    text = URL_RE.sub("<url>", text)
    if strip_emoji:
        text = EMOJI_RE.sub(" ", text)
    if fix_elongation:
        text = ELONGATION_RE.sub(r"\1\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()

def fetch_raw(force=False):
    """download the three tsv files once and keep them on disk"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, filename in SPLIT_FILES.items():
        dest = RAW_DIR / f"{split}.tsv"
        if force or not dest.exists():
            urllib.request.urlretrieve(f"{GITHUB_BASE}/{filename}", dest)
        paths[split] = dest
    return paths

def load_splits(strip_emoji=False, fix_elongation=False):
    """gives back the three splits with text raw_text and label columns"""
    out = {}
    for split, path in fetch_raw().items():
        # quoting=3 stops a stray quote mark eating the rows after it
        df = pd.read_csv(path, sep="\t", quoting=3, keep_default_na=False)
        df = df.rename(columns={"tweet": "raw_text"})
        df["label"] = df["label"].map(LABEL_TO_ID).astype(int)
        df["text"] = df["raw_text"].map(
            lambda t: clean_text(t, strip_emoji, fix_elongation))
        df = df[df["text"].str.len() > 0].reset_index(drop=True)
        out[split] = df[["text", "raw_text", "label"]]
        print(f"  {split}: {len(out[split])} tweets")
    return out

def describe(splits):
    """counts tweets per class and measures how long they run"""
    rows = []
    for split, df in splits.items():
        lengths = df["text"].str.split().str.len()
        row = {"split": split, "n": len(df),
               "mean_tokens": round(lengths.mean(), 1),
               "median_tokens": int(lengths.median()),
               "p95_tokens": int(lengths.quantile(0.95)),
               "max_tokens": int(lengths.max())}
        counts = df["label"].value_counts()
        for label_id, name in ID_TO_LABEL.items():
            n = int(counts.get(label_id, 0))
            row[f"n_{name}"] = n
            row[f"pct_{name}"] = round(100 * n / len(df), 1)
        rows.append(row)
    return pd.DataFrame(rows)

def save_processed(splits):
    """saving the cleaned splits so later steps do not download again"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split, df in splits.items():
        df.to_csv(PROCESSED_DIR / f"{split}.csv", index=False)
    with open(PROCESSED_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(ID_TO_LABEL, f, indent=2)

def load_processed():
    """reads back whatever save_processed is going toput on disk"""
    return {s: pd.read_csv(PROCESSED_DIR / f"{s}.csv")
            for s in ("train", "validation", "test")}

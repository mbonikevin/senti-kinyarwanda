"""check whether the official splits share tweets or disagree on labels"""

import pandas as pd

from src.data import ID_TO_LABEL

def overlap_report(splits):
    """counts how many tweets each pair of splits shares and where labels clash"""
    rows = []
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        da, db = splits[a], splits[b]
        shared = set(da["text"]) & set(db["text"])
        clashes = 0
        if shared:
            la = da.drop_duplicates("text").set_index("text")["label"]
            lb = db.drop_duplicates("text").set_index("text")["label"]
            clashes = sum(la[t] != lb[t] for t in shared)
        rows.append({"split_a": a, "split_b": b, "shared_texts": len(shared),
                     "contradictory_labels": clashes,
                     "pct_of_b": round(100 * len(shared) / len(db), 2)})
    return pd.DataFrame(rows)

def contradictory_examples(splits, a="train", b="test"):
    """tweets with the same text but a different label in each split"""
    da, db = splits[a], splits[b]
    shared = set(da["text"]) & set(db["text"])
    la = da.drop_duplicates("text").set_index("text")["label"]
    lb = db.drop_duplicates("text").set_index("text")["label"]
    rows = [{"text": t, f"label_in_{a}": ID_TO_LABEL[la[t]],
             f"label_in_{b}": ID_TO_LABEL[lb[t]]}
            for t in shared if la[t] != lb[t]]
    return pd.DataFrame(rows)

def clean_mask(splits, split="test"):
    """marks true when the tweet is not sitting in training as well"""
    return ~splits[split]["text"].isin(set(splits["train"]["text"]))

def annotation_ceiling(splits, split="test"):
    """the best accuracy anyone can reach because clashing labels cannot both hold"""
    clashes = len(contradictory_examples(splits, "train", split))
    n = len(splits[split])
    return {"split": split, "n": n, "contradictory_rows": clashes,
            "accuracy_ceiling": round(1 - clashes / n, 4)}

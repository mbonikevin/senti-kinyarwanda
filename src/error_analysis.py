"""find which tweets the model gets wrong and what they have in common"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ID_TO_LABEL

RESULTS_DIR = Path("reports/results")

# common english words to spot tweets that mix english into kinyarwanda
ENGLISH_MARKERS = {
    "the", "this", "that", "and", "but", "for", "with", "you", "your", "they",
    "have", "has", "will", "what", "when", "why", "how", "not", "very", "just",
    "please", "thank", "thanks", "god", "people", "time", "good", "about", "all",
    "from", "there", "here", "now", "today", "more", "than", "then", "some", "any",
    "can", "was", "were", "are", "been", "being", "which", "who", "whom", "our",
}
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]")

FEATURES = ["very_short", "has_emoji", "has_english", "has_url",
            "mentions_only", "has_question", "has_elongation"]

def tag_features(text):
    """marks surface traits that may be making a tweet hard to call"""
    # 8 tokens not 5 because nothing in this data is shorter than 5 words
    stripped = re.sub(r"@user|<url>|https?://\S+", " ", text).strip()
    tokens = text.split()
    words = {w.strip(".,!?;:\"'()").lower() for w in tokens}
    return {"n_tokens": len(tokens),
"very_short": len(tokens) < 8,
            "has_emoji": bool(EMOJI_RE.search(text)),
            "has_english": len(words & ENGLISH_MARKERS) >= 1,
            "has_url": "<url>" in text or "http" in text,
            "mentions_only": len(stripped.split()) < 5,
            "has_question": "?" in text,
            "has_elongation": bool(re.search(r"(.)\1{2,}", text))}

def build_frame(df, preds, probs, model_name="model"):
    """one row per tweet with the true label the guess confidence and tags"""
    out = df.reset_index(drop=True).copy()
    out["model"] = model_name
    out["pred"] = preds
    out["true_label"] = out["label"].map(ID_TO_LABEL)
    out["pred_label"] = pd.Series(preds).map(ID_TO_LABEL)
    out["correct"] = out["label"] == out["pred"]
    out["confidence"] = probs.max(axis=1).round(4)
    out["true_prob"] = probs[np.arange(len(out)), out["label"].to_numpy()].round(4)
    return pd.concat([out, pd.DataFrame([tag_features(t) for t in out["text"]])], axis=1)

def feature_error_rates(frame):
    """compares error rates with and without each tag a lift near 1 says nothing is there"""
    overall = 1 - frame["correct"].mean()
    rows = []
    for feat in FEATURES:
        with_f, without_f = frame[frame[feat]], frame[~frame[feat]]
        if len(with_f) == 0:
            continue
        err_with = 1 - with_f["correct"].mean()
        err_without = 1 - without_f["correct"].mean() if len(without_f) else float("nan")
        rows.append({"feature": feat, "n_with": len(with_f),
                     "err_with": round(err_with, 4), "err_without": round(err_without, 4),
                     "lift": round(err_with / err_without, 2) if err_without else float("nan")})
    print(f"Overall error rate: {overall:.4f} on {len(frame)} tweets")
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)

def confusion_pairs(frame):
    """shows which class gets taken for which most common first"""
    wrong = frame[~frame["correct"]]
    pairs = (wrong.groupby(["true_label", "pred_label"]).size()
             .reset_index(name="n").sort_values("n", ascending=False))
    pairs["pct_of_errors"] = (100 * pairs["n"] / len(wrong)).round(1)
    return pairs.reset_index(drop=True)

def export(frame, model_name):
    """save errors tag rates confusion pairs and the best correct guesses"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.lower().replace(" ", "_").replace("/", "_")
    cols = ["raw_text", "true_label", "pred_label", "confidence", "true_prob",
            "n_tokens"] + FEATURES

    # sorted by confidence a confident mistake shows what the model learned wrong
    paths = {"errors": RESULTS_DIR / f"errors_{safe}.csv",
             "feature_rates": RESULTS_DIR / f"error_features_{safe}.csv",
             "confusion_pairs": RESULTS_DIR / f"confusion_pairs_{safe}.csv",
             "top_correct": RESULTS_DIR / f"top_correct_{safe}.csv"}
    frame[~frame["correct"]].sort_values("confidence", ascending=False)[cols] \
        .to_csv(paths["errors"], index=False)
    feature_error_rates(frame).to_csv(paths["feature_rates"], index=False)
    confusion_pairs(frame).to_csv(paths["confusion_pairs"], index=False)
    frame[frame["correct"]].sort_values("confidence", ascending=False).head(20)[cols] \
        .to_csv(paths["top_correct"], index=False)
    return paths

def disagreement(frame_a, frame_b, name_a="baseline", name_b="transformer"):
    """lines up the tweets the two models answer differently and says who wins"""
    if len(frame_a) != len(frame_b):
        raise ValueError("frames must cover the same split in the same order")
    out = pd.DataFrame({
        "raw_text": frame_a["raw_text"].to_numpy(),
        "true_label": frame_a["true_label"].to_numpy(),
        f"pred_{name_a}": frame_a["pred_label"].to_numpy(),
        f"pred_{name_b}": frame_b["pred_label"].to_numpy(),
        f"correct_{name_a}": frame_a["correct"].to_numpy(),
        f"correct_{name_b}": frame_b["correct"].to_numpy()})
    d = out[out[f"pred_{name_a}"] != out[f"pred_{name_b}"]].copy()
    d["winner"] = np.where(d[f"correct_{name_b}"], name_b,
                           np.where(d[f"correct_{name_a}"], name_a, "neither"))
    print(d["winner"].value_counts().to_string())
    return d.reset_index(drop=True)

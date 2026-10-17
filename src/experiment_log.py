"""save one row per training run with the guess made before it ran"""

import csv
from datetime import date
from pathlib import Path

LOG_DIR = Path("reports/results")
# runs closer than this on validation count as a tie so pick the cheaper one
TIE_THRESHOLD = 0.01

COLUMNS = ["exp_id", "date", "model", "stage", "change", "hypothesis",
           "learning_rate", "epochs", "max_len", "batch_size", "seed",
           "val_macro_f1", "val_weighted_f1", "val_accuracy",
           "train_seconds", "decision", "reason"]

def log_path(model):
    """file that holds one model's runs"""
    return LOG_DIR / f"experiments_{model.lower().replace(' ', '_').replace('/', '_')}.csv"

def log_experiment(**row):
    """adds one run to the log everything but date has to be filled in"""
    missing = [c for c in COLUMNS if c not in row and c != "date"]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    row.setdefault("date", date.today().isoformat())

    path = log_path(row["model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in COLUMNS})

def reset_log(model):
    """delete a model's log so rerunning does not add the same rows twice"""
    log_path(model).unlink(missing_ok=True)

def load_all_experiments():
    """joins every model log into a single table"""
    import pandas as pd

    files = sorted(LOG_DIR.glob("experiments_*.csv"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values(["model", "exp_id"]).reset_index(drop=True)

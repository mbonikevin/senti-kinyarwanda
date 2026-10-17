"""finetune afroxlmr on the tweets and log each run"""

import time
from dataclasses import dataclass

import numpy as np

from src.data import ID_TO_LABEL, LABEL_ORDER
from src.metrics import compute_metrics

AFRO_XLMR_BASE = "Davlan/afro-xlmr-base"
XLMR_BASE = "FacebookAI/xlm-roberta-base"
AFRIBERTA_BASE = "castorini/afriberta_base"

@dataclass
class TrainConfig:
    """holds one training setup the experiments move these numbers around"""
    model_name: str = AFRO_XLMR_BASE
    learning_rate: float = 2e-5
    epochs: int = 3
    max_len: int = 128
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    class_weights: bool = False
    label: str = "afroxlmr-default"

    def summary(self):
        return (f"{self.model_name.split('/')[-1]} lr={self.learning_rate:g} "
                f"ep={self.epochs} len={self.max_len} bs={self.batch_size} "
                f"cw={self.class_weights} seed={self.seed}")

def _cuda_available():
    import torch
    return torch.cuda.is_available()

def _training_args(cfg, out_dir):
    """build the training settings handling an argument renamed in transformers 441"""
    from transformers import TrainingArguments

    common = dict(output_dir=out_dir, learning_rate=cfg.learning_rate,
                  per_device_train_batch_size=cfg.batch_size,
                  per_device_eval_batch_size=64, num_train_epochs=cfg.epochs,
                  weight_decay=cfg.weight_decay, warmup_ratio=cfg.warmup_ratio,
                  seed=cfg.seed, logging_steps=50, save_total_limit=1, report_to=[],
                  fp16=_cuda_available(), load_best_model_at_end=True,
                  metric_for_best_model="macro_f1", greater_is_better=True)
    try:
        return TrainingArguments(eval_strategy="epoch", save_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", save_strategy="epoch", **common)

def _hf_metrics(eval_pred):
    """turns raw model output into scores the trainer prints each epoch"""
    logits, labels = eval_pred
    m = compute_metrics(labels, np.argmax(logits, axis=-1))
    return {k: m[k] for k in ("accuracy", "macro_f1", "macro_precision",
                              "macro_recall", "weighted_f1")}

def _tokenize(splits, tokenizer, max_len):
    """cuts text into tokens padding waits for the batch so we do not waste room"""
    from datasets import Dataset

    out = {}
    for split, df in splits.items():
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        out[split] = ds.map(lambda b: tokenizer(b["text"], truncation=True,
                                                max_length=max_len),
                            batched=True, remove_columns=["text"])
    return out

def _class_weight_tensor(train_df):
    """hands the rarer classes more weight the way scikit learn does it"""
    import torch

    counts = np.array([(train_df["label"] == i).sum() for i in range(len(LABEL_ORDER))],
                      dtype=float)
    return torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)),
                        dtype=torch.float)

def _weighted_trainer_class(weights):
    """trainer that applies class weights since the default loss cannot"""
    from torch import nn
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))(
                outputs.logits.view(-1, len(LABEL_ORDER)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer

def train(cfg, splits, out_dir=None):
    """finetune one setup and keep the best epoch returns trainer tokenizer seconds"""
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, set_seed)

    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # id2label goes into the saved config so the app shows real class names
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(LABEL_ORDER), id2label=ID_TO_LABEL,
        label2id={v: k for k, v in ID_TO_LABEL.items()})

    tokenized = _tokenize(splits, tokenizer, cfg.max_len)
    TrainerCls = (_weighted_trainer_class(_class_weight_tensor(splits["train"]))
                  if cfg.class_weights else Trainer)
    trainer = TrainerCls(
        model=model, args=_training_args(cfg, out_dir or f"checkpoints/{cfg.label}"),
        train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=_hf_metrics)

    start = time.time()
    trainer.train()
    return trainer, tokenizer, time.time() - start

def predict(trainer, splits, split, max_len, tokenizer):
    """returns predicted labels and class chances for one split"""
    import torch

    output = trainer.predict(_tokenize({split: splits[split]}, tokenizer, max_len)[split])
    probs = torch.softmax(torch.tensor(output.predictions), dim=-1).numpy()
    return np.argmax(output.predictions, axis=-1), probs

def run_experiment(exp_id, cfg, splits, change, hypothesis, decision="", reason=""):
    """trains one setup scores it on validation only then writes the row"""
    from src.experiment_log import log_experiment

    print(f"\n=== {exp_id}: {cfg.summary()} ===\n{change}")
    trainer, tokenizer, seconds = train(cfg, splits)
    preds, _ = predict(trainer, splits, "validation", cfg.max_len, tokenizer)
    m = compute_metrics(splits["validation"]["label"], preds)

    log_experiment(exp_id=exp_id, model=cfg.model_name.split("/")[-1], stage=cfg.label,
                   change=change, hypothesis=hypothesis,
                   learning_rate=cfg.learning_rate, epochs=cfg.epochs,
                   max_len=cfg.max_len, batch_size=cfg.batch_size, seed=cfg.seed,
                   val_macro_f1=round(m["macro_f1"], 4),
                   val_weighted_f1=round(m["weighted_f1"], 4),
                   val_accuracy=round(m["accuracy"], 4),
                   train_seconds=round(seconds, 1), decision=decision, reason=reason)
    print(f"val macro-F1 {m['macro_f1']:.4f} | weighted-F1 {m['weighted_f1']:.4f} "
          f"| acc {m['accuracy']:.4f} | {seconds:.0f}s")
    return {"exp_id": exp_id, "cfg": cfg, "trainer": trainer,
            "tokenizer": tokenizer, "metrics": m, "seconds": seconds}

"""measure how the tokenizers cut kinyarwanda words into pieces"""

import numpy as np
import pandas as pd

CANDIDATES = {
    "afro-xlmr-base": "Davlan/afro-xlmr-base",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "afriberta_base": "castorini/afriberta_base",
}

def fertility_table(texts, max_len=128, candidates=None):
    """pieces per word and sequence lengths this is what picked max_len"""
    from transformers import AutoTokenizer

    candidates = candidates or CANDIDATES
    n_words = sum(len(t.split()) for t in texts)
    rows = []
    for name, model_id in candidates.items():
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:
            print(f"  skipped {name}: {type(exc).__name__}")
            continue
        lengths = np.array([len(tok(t)["input_ids"]) for t in texts])
        rows.append({"tokenizer": name, "vocab_size": tok.vocab_size,
                     "subwords_per_word": round(lengths.sum() / n_words, 2),
                     "median_len": int(np.median(lengths)),
                     "p95_len": int(np.percentile(lengths, 95)),
                     "max_len_seen": int(lengths.max()),
                     f"pct_truncated_at_{max_len}": round(100 * (lengths > max_len).mean(), 2)})
    return pd.DataFrame(rows)

def segmentation_examples(model_id="Davlan/afro-xlmr-base", words=None):
    """show how single words are split so the numbers above feel real"""
    from transformers import AutoTokenizer

    words = words or ["Abanyarwanda", "ntabyihogoza", "byagutabara",
                      "isobanurampamvu", "umuryango", "good"]
    tok = AutoTokenizer.from_pretrained(model_id)
    return pd.DataFrame([{"word": w, "n_subwords": len(tok.tokenize(w)),
                          "segmentation": " | ".join(tok.tokenize(w))} for w in words])

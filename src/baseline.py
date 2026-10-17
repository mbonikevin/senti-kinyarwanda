"""simple models the transformer has to beat"""

import time

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from src.metrics import compute_metrics

def build_majority():
    """this one just picks whichever class showed up most in training"""
    return DummyClassifier(strategy="most_frequent")

def build_nb():
    """word counts into naive bayes"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", MultinomialNB()),
    ])

def build_logreg():
    """feeds word and letter patterns into logistic regression"""
    return Pipeline([
        # kinyarwanda sticks prefixes and suffixes onto a stem so letter
        # patterns catch forms that a word only model walks straight past
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                       min_df=2, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])

BASELINES = {
    "Majority class": build_majority,
    "TF-IDF + NaiveBayes": build_nb,
    "TF-IDF + LogReg": build_logreg,
}

def run_baseline(name, splits, eval_split="validation"):
    """train one baseline and score it returns the model scores and seconds"""
    model = BASELINES[name]()
    start = time.time()
    model.fit(splits["train"]["text"], splits["train"]["label"])
    elapsed = time.time() - start
    preds = model.predict(splits[eval_split]["text"])
    return model, compute_metrics(splits[eval_split]["label"], preds), elapsed

def top_features(pipeline, n=15):
    """pulls the strongest words per class so we can see what it picked up"""
    import numpy as np

    from src.data import LABEL_ORDER

    word_vec = dict(pipeline.named_steps["features"].transformer_list)["word"]
    names = np.array(word_vec.get_feature_names_out())
    coefs = pipeline.named_steps["clf"].coef_[:, :len(names)]
    return {label: [(names[j], round(float(coefs[i][j]), 3))
                    for j in np.argsort(coefs[i])[::-1][:n]]
            for i, label in enumerate(LABEL_ORDER)}

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from csrc_rag.training.data import build_label_vocab, encode_multilabel
from csrc_rag.training.metrics import multilabel_metrics


@dataclass(frozen=True)
class SklearnTrainingOutput:
    metrics: dict[str, float]
    label_vocab: list[str]


def train_tfidf_logreg(
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
) -> SklearnTrainingOutput:
    vocab = build_label_vocab(train_rows + valid_rows + test_rows)
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

    x_train = vectorizer.fit_transform([row["input_text"] for row in train_rows])
    x_test = vectorizer.transform([row["input_text"] for row in test_rows])

    y_train = encode_multilabel(train_rows, vocab)
    y_test = encode_multilabel(test_rows, vocab)

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="liblinear")
    )
    classifier.fit(x_train, y_train)
    y_prob = classifier.predict_proba(x_test)
    y_pred = (y_prob >= 0.5).astype(np.float32)
    metrics = multilabel_metrics(y_test, y_pred)
    return SklearnTrainingOutput(metrics=metrics, label_vocab=vocab)


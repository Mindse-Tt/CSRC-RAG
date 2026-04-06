from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from csrc_rag.settings import ARTIFACTS_DIR, CONFIG_DIR


DEFAULT_INTENT_ARTIFACT = ARTIFACTS_DIR / "intent_classifier" / "intent_model.pkl"
DEFAULT_INTENT_REPORT = ARTIFACTS_DIR / "intent_classifier" / "intent_report.json"


@dataclass(frozen=True)
class IntentPrediction:
    name: str
    confidence: float
    scores: dict[str, float]
    method: str


class TfidfIntentClassifier:
    def __init__(self, vectorizer: TfidfVectorizer, classifier: LogisticRegression, labels: list[str]) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.labels = labels

    def predict(self, query: str) -> IntentPrediction:
        features = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(features)[0]
        scores = {
            label: float(probability)
            for label, probability in sorted(
                zip(self.classifier.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        label = max(scores.items(), key=lambda item: item[1])[0]
        return IntentPrediction(
            name=label,
            confidence=round(scores[label], 4),
            scores={key: round(value, 4) for key, value in scores.items()},
            method="tfidf_logistic_regression",
        )


def load_examples(path: str | Path | None = None) -> tuple[list[str], list[str]]:
    config_path = Path(path) if path else CONFIG_DIR / "intent_examples.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    texts: list[str] = []
    labels: list[str] = []
    for label, examples in payload.items():
        for text in examples:
            texts.append(text)
            labels.append(label)
    return texts, labels


def _augment_texts(texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    prefixes = ["", "请帮我", "帮我", "请问", "我想知道"]
    suffixes = ["", "。", "，请解释一下。", "，用于课程项目。"]
    augmented_texts: list[str] = []
    augmented_labels: list[str] = []
    for text, label in zip(texts, labels):
        for prefix in prefixes:
            for suffix in suffixes:
                candidate = f"{prefix}{text}{suffix}".strip()
                augmented_texts.append(candidate)
                augmented_labels.append(label)
    return augmented_texts, augmented_labels


def train_intent_classifier(
    examples_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    texts, labels = load_examples(examples_path)
    texts, labels = _augment_texts(texts, labels)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    classifier = LogisticRegression(max_iter=2000, C=4.0)
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)
    classifier.fit(x_train_features, y_train)
    predictions = classifier.predict(x_test_features)
    report = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "num_train_examples": len(x_train),
        "num_test_examples": len(x_test),
        "labels": sorted(set(labels)),
    }

    model = TfidfIntentClassifier(vectorizer=vectorizer, classifier=classifier, labels=sorted(set(labels)))
    output_path = Path(artifact_path) if artifact_path else DEFAULT_INTENT_ARTIFACT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)

    output_report = Path(report_path) if report_path else DEFAULT_INTENT_REPORT
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_intent_classifier(path: str | Path | None = None) -> TfidfIntentClassifier | None:
    model_path = Path(path) if path else DEFAULT_INTENT_ARTIFACT
    if not model_path.exists():
        return None
    with model_path.open("rb") as handle:
        return pickle.load(handle)

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

UNKNOWN_THRESHOLD = 0.20


def evaluate_predictions(y_true, y_pred):

    results = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision_weighted": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    return results


def print_classification_report(
    y_true,
    y_pred
):
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


def calculate_confidence_margin(
    decision_scores
):
    """
    Calculate the difference between the
    highest and second-highest SVM decision scores.
    """

    sorted_scores = np.sort(
        decision_scores,
        axis=1
    )

    top_score = sorted_scores[:, -1]
    second_score = sorted_scores[:, -2]

    margin = (
        top_score - second_score
    )

    return margin


def classify_ticket_status(
    margin,
    threshold=UNKNOWN_THRESHOLD
):
    if margin < threshold:
        return "UNKNOWN / REVIEW"

    return "KNOWN / ROUTE"


def route_tickets(
    texts,
    model=None,
    vectorizer=None,
    threshold=UNKNOWN_THRESHOLD
):

    if model is None:
        model = joblib.load(
            MODELS_DIR / "best_svm_model.pkl"
        )

    if vectorizer is None:
        vectorizer = joblib.load(
            MODELS_DIR
            / "best_tfidf_vectorizer.pkl"
        )

    X_tfidf = vectorizer.transform(
        texts
    )

    predictions = model.predict(
        X_tfidf
    )

    decision_scores = (
        model.decision_function(
            X_tfidf
        )
    )

    margins = calculate_confidence_margin(
        decision_scores
    )

    results = pd.DataFrame({
        "text": texts,
        "predicted_queue": predictions,
        "confidence_margin": margins
    })

    results["status"] = results[
        "confidence_margin"
    ].apply(
        lambda value: classify_ticket_status(
            value,
            threshold
        )
    )

    return results


if __name__ == "__main__":

    model = joblib.load(
        MODELS_DIR / "best_svm_model.pkl"
    )

    vectorizer = joblib.load(
        MODELS_DIR
        / "best_tfidf_vectorizer.pkl"
    )

    print(
        "SVM model loaded."
    )

    print(
        "TF-IDF vectorizer loaded."
    )

    print(
        "Number of known classes:",
        len(model.classes_)
    )

    print(
        "\nKnown queues:"
    )

    print(model.classes_)
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.data.load_data import load_train_split, load_test_split
from src.features.tfidf_features import create_baseline_tfidf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def train_logistic_regression():

    train_df = load_train_split()
    test_df = load_test_split()

    X_train = train_df["clean_text"]
    y_train = train_df["queue"]

    X_test = test_df["clean_text"]
    y_test = test_df["queue"]

    # Baseline TF-IDF
    tfidf = create_baseline_tfidf()

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Logistic Regression
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    logistic_model.fit(
        X_train_tfidf,
        y_train
    )

    y_pred = logistic_model.predict(
        X_test_tfidf
    )

    results = {
        "model": "Logistic Regression",
        "features": "TF-IDF",
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "precision_weighted": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        tfidf,
        MODELS_DIR / "tfidf_vectorizer.pkl"
    )

    joblib.dump(
        logistic_model,
        MODELS_DIR / "logistic_regression.pkl"
    )

    return logistic_model, tfidf, results


if __name__ == "__main__":
    model, vectorizer, results = (
        train_logistic_regression()
    )

    print(results)
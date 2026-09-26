from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

SPLITS_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


def load_training_data():

    train_df = pd.read_csv(
        SPLITS_DIR / "train.csv"
    )

    test_df = pd.read_csv(
        SPLITS_DIR / "test.csv"
    )

    X_train = train_df["clean_text"]
    y_train = train_df["queue"]

    X_test = test_df["clean_text"]
    y_test = test_df["queue"]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def evaluate_model(y_test, predictions):

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }


def create_svm():

    return LinearSVC(
        C=10,
        class_weight="balanced",
        random_state=42
    )


def run_svm_experiments():

    X_train, X_test, y_train, y_test = (
        load_training_data()
    )

    results = []

    # ------------------------------------------------
    # Experiment 1: Word TF-IDF (1,1)
    # ------------------------------------------------

    word_11 = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_11 = word_11.fit_transform(
        X_train
    )

    X_test_11 = word_11.transform(
        X_test
    )

    svm_11 = create_svm()

    svm_11.fit(
        X_train_11,
        y_train
    )

    pred_11 = svm_11.predict(
        X_test_11
    )

    result_11 = evaluate_model(
        y_test,
        pred_11
    )

    result_11["Model"] = (
        "SVM + Word TF-IDF (1,1)"
    )

    results.append(result_11)

    # ------------------------------------------------
    # Experiment 2: Word TF-IDF (1,2)
    # ------------------------------------------------

    word_12 = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_12 = word_12.fit_transform(
        X_train
    )

    X_test_12 = word_12.transform(
        X_test
    )

    svm_12 = create_svm()

    svm_12.fit(
        X_train_12,
        y_train
    )

    pred_12 = svm_12.predict(
        X_test_12
    )

    result_12 = evaluate_model(
        y_test,
        pred_12
    )

    result_12["Model"] = (
        "SVM + Word TF-IDF (1,2)"
    )

    results.append(result_12)

    # ------------------------------------------------
    # Experiment 3: Word TF-IDF (1,3)
    # ------------------------------------------------

    word_13 = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_13 = word_13.fit_transform(
        X_train
    )

    X_test_13 = word_13.transform(
        X_test
    )

    svm_13 = create_svm()

    svm_13.fit(
        X_train_13,
        y_train
    )

    pred_13 = svm_13.predict(
        X_test_13
    )

    result_13 = evaluate_model(
        y_test,
        pred_13
    )

    result_13["Model"] = (
        "SVM + Word TF-IDF (1,3)"
    )

    results.append(result_13)

    # ------------------------------------------------
    # Experiment 4: Character TF-IDF (3,5)
    # ------------------------------------------------

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_char = char_tfidf.fit_transform(
        X_train
    )

    X_test_char = char_tfidf.transform(
        X_test
    )

    svm_char = create_svm()

    svm_char.fit(
        X_train_char,
        y_train
    )

    pred_char = svm_char.predict(
        X_test_char
    )

    result_char = evaluate_model(
        y_test,
        pred_char
    )

    result_char["Model"] = (
        "SVM + Character TF-IDF (3,5)"
    )

    results.append(result_char)

    # ------------------------------------------------
    # Experiment 5:
    # Word (1,2) + Character TF-IDF
    # ------------------------------------------------

    X_train_combined = hstack([
        X_train_12,
        X_train_char
    ])

    X_test_combined = hstack([
        X_test_12,
        X_test_char
    ])

    svm_combined = create_svm()

    svm_combined.fit(
        X_train_combined,
        y_train
    )

    pred_combined = svm_combined.predict(
        X_test_combined
    )

    result_combined = evaluate_model(
        y_test,
        pred_combined
    )

    result_combined["Model"] = (
        "SVM + Word (1,2) + Character TF-IDF"
    )

    results.append(result_combined)

    results_df = pd.DataFrame(results)

    # Reorder columns
    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "Macro F1",
            "Weighted F1"
        ]
    ]

    return results_df


def train_final_svm():

    X_train, X_test, y_train, y_test = (
        load_training_data()
    )

    # Final selected Word TF-IDF (1,3)
    best_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_best = best_tfidf.fit_transform(
        X_train
    )

    X_test_best = best_tfidf.transform(
        X_test
    )

    # Final SVM
    best_svm = LinearSVC(
        C=10,
        class_weight="balanced",
        random_state=42
    )

    best_svm.fit(
        X_train_best,
        y_train
    )

    best_pred = best_svm.predict(
        X_test_best
    )

    print(
        classification_report(
            y_test,
            best_pred,
            zero_division=0
        )
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_tfidf,
        MODELS_DIR / "best_tfidf_vectorizer.pkl"
    )

    joblib.dump(
        best_svm,
        MODELS_DIR / "best_svm_model.pkl"
    )

    return (
        best_svm,
        best_tfidf,
        y_test,
        best_pred
    )


if __name__ == "__main__":

    comparison = run_svm_experiments()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        REPORTS_DIR / "tfidf_svm_experiments.csv",
        index=False
    )

    print(comparison)

    train_final_svm()

    print("Best TF-IDF vectorizer saved.")
    print("Best SVM model saved.")
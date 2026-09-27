from pathlib import Path
import mlflow
import joblib
import pandas as pd
import mlflow.sklearn


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
        random_state=42,
        max_iter=5000
    )


def run_svm_experiments():
    mlflow.set_experiment(
        "adaptive-ticket-routing-svm"
    )

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

    with mlflow.start_run(run_name="svm_word_1_1"):
        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("C", 10)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_type", "word_tfidf")
        mlflow.log_param("ngram_range", "(1,1)")

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

        mlflow.log_metrics({
            "accuracy": result_11["Accuracy"],
            "precision": result_11["Precision"],
            "recall": result_11["Recall"],
            "macro_f1": result_11["Macro F1"],
            "weighted_f1": result_11["Weighted F1"]
        })

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

    with mlflow.start_run(run_name="svm_word_1_2"):
        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("C", 10)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_type", "word_tfidf")
        mlflow.log_param("ngram_range", "(1,2)")
        
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
        
        mlflow.log_metrics({
            "accuracy": result_12["Accuracy"],
            "precision": result_12["Precision"],
            "recall": result_12["Recall"],
            "macro_f1": result_12["Macro F1"],
            "weighted_f1": result_12["Weighted F1"]
        })

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
    
    with mlflow.start_run(run_name="svm_word_1_3"):
        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("C", 10)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_type", "word_tfidf")
        mlflow.log_param("ngram_range", "(1,3)")

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
        
        mlflow.log_metrics({
            "accuracy": result_13["Accuracy"],
            "precision": result_13["Precision"],
            "recall": result_13["Recall"],
            "macro_f1": result_13["Macro F1"],
            "weighted_f1": result_13["Weighted F1"]
        })

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
    
    with mlflow.start_run(run_name="svm_character_3_5"):
        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("C", 10)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_type", "character_tfidf")
        mlflow.log_param("ngram_range", "(3,5)")

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
        
        mlflow.log_metrics({
            "accuracy": result_char["Accuracy"],
            "precision": result_char["Precision"],
            "recall": result_char["Recall"],
            "macro_f1": result_char["Macro F1"],
            "weighted_f1": result_char["Weighted F1"]
        })

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

    with mlflow.start_run(run_name="svm_word_1_2_character"):

        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("C", 10)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("feature_type", "word_character_combined")
        mlflow.log_param("word_ngram_range", "(1,2)")
        mlflow.log_param("character_ngram_range", "(3,5)")
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
        
        mlflow.log_metrics({
            "accuracy": result_combined["Accuracy"],
            "precision": result_combined["Precision"],
            "recall": result_combined["Recall"],
            "macro_f1": result_combined["Macro F1"],
            "weighted_f1": result_combined["Weighted F1"]
        })

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
    best_svm = create_svm()

    # Start MLflow run for the final model
    mlflow.set_experiment(
        "adaptive-ticket-routing-final"
    )

    with mlflow.start_run(
        run_name="final_svm_word_tfidf_1_3"
    ):

        # Log model parameters
        mlflow.log_param(
            "model",
            "LinearSVC"
        )

        mlflow.log_param(
            "C",
            10
        )

        mlflow.log_param(
            "class_weight",
            "balanced"
        )

        mlflow.log_param(
            "feature_type",
            "word_tfidf"
        )

        mlflow.log_param(
            "ngram_range",
            "(1,3)"
        )

        mlflow.log_param(
            "min_df",
            2
        )

        mlflow.log_param(
            "max_df",
            0.95
        )

        mlflow.log_param(
            "sublinear_tf",
            True
        )

        # Train final model
        best_svm.fit(
            X_train_best,
            y_train
        )

        # Predictions
        best_pred = best_svm.predict(
            X_test_best
        )

        # Calculate metrics
        final_metrics = evaluate_model(
            y_test,
            best_pred
        )

        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": final_metrics["Accuracy"],
            "precision": final_metrics["Precision"],
            "recall": final_metrics["Recall"],
            "macro_f1": final_metrics["Macro F1"],
            "weighted_f1": final_metrics["Weighted F1"]
        })

        # Log final SVM model to MLflow
        mlflow.sklearn.log_model(
            sk_model=best_svm,
            artifact_path="svm_model"
        )

        # Log TF-IDF vectorizer separately
        mlflow.sklearn.log_model(
            sk_model=best_tfidf,
            artifact_path="tfidf_vectorizer"
        )

        print(
            "\nFinal model logged to MLflow."
        )

        print(
            f"MLflow Weighted F1: "
            f"{final_metrics['Weighted F1']:.4f}"
        )

        print(
            f"MLflow Accuracy: "
            f"{final_metrics['Accuracy']:.4f}"
        )

    # Print classification report
    print(
        classification_report(
            y_test,
            best_pred,
            zero_division=0
        )
    )

    # Save local artifacts
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
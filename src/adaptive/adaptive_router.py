from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_svm_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "best_tfidf_vectorizer.pkl"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"


# ---------------------------------------------------------
# NOVELTY SETTINGS
# ---------------------------------------------------------

# Minimum difference between the best and second-best
# SVM class score.
MARGIN_THRESHOLD = 0.10

# Minimum similarity to examples from the predicted class.
# This is deliberately lower because TF-IDF similarity
# can be small even for valid tickets.
SIMILARITY_THRESHOLD = 0.15


def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def load_training_examples(vectorizer):

    train_df = pd.read_csv(TRAIN_PATH)

    train_text = (
        train_df["clean_text"]
        .fillna("")
        .astype(str)
    )

    train_X = vectorizer.transform(train_text)

    return train_df, train_X


def calculate_class_similarity(
    ticket_vector,
    predicted_class,
    train_df,
    train_X,
):
    """
    Find the most similar training ticket belonging
    to the predicted category.

    TF-IDF vectors are normalized, so dot product
    corresponds to cosine similarity.
    """

    class_mask = (
        train_df["queue"].values == predicted_class
    )

    if not np.any(class_mask):
        return 0.0

    class_vectors = train_X[class_mask]

    similarities = (
        class_vectors @ ticket_vector.T
    ).toarray().ravel()

    if len(similarities) == 0:
        return 0.0

    return float(np.max(similarities))


def predict_ticket(text):

    model, vectorizer = load_model()

    text = str(text).strip()

    if not text:
        raise ValueError(
            "Ticket text cannot be empty."
        )

    # -----------------------------------------------------
    # Transform ticket
    # -----------------------------------------------------

    X = vectorizer.transform([text])

    # -----------------------------------------------------
    # SVM prediction
    # -----------------------------------------------------

    prediction = model.predict(X)[0]

    scores = model.decision_function(X)

    # -----------------------------------------------------
    # Handle binary/multiclass SVM
    # -----------------------------------------------------

    if scores.ndim == 1:

        # Multiclass LinearSVC normally returns
        # one score per class.
        class_scores = scores

    else:

        class_scores = scores[0]

    # -----------------------------------------------------
    # Best and second-best class
    # -----------------------------------------------------

    sorted_scores = np.sort(
        class_scores
    )[::-1]

    best_score = float(
        sorted_scores[0]
    )

    if len(sorted_scores) > 1:

        second_score = float(
            sorted_scores[1]
        )

    else:

        second_score = best_score

    # Difference between best and second-best.
    margin = (
        best_score
        - second_score
    )

    # -----------------------------------------------------
    # Compare against real training examples
    # -----------------------------------------------------

    train_df, train_X = (
        load_training_examples(vectorizer)
    )

    similarity = calculate_class_similarity(
        X,
        prediction,
        train_df,
        train_X,
    )

    # -----------------------------------------------------
    # NOVELTY DECISION
    # -----------------------------------------------------

    is_outsider = (
        margin < MARGIN_THRESHOLD
        or similarity < SIMILARITY_THRESHOLD
    )

    # -----------------------------------------------------
    # Outsider
    # -----------------------------------------------------

    if is_outsider:

        return {
            "status": "OUTSIDER",
            "predicted_queue": None,
            "confidence": round(
                margin,
                4
            ),
            "similarity": round(
                similarity,
                4
            ),
        }

    # -----------------------------------------------------
    # Known category
    # -----------------------------------------------------

    return {
        "status": "KNOWN",
        "predicted_queue": prediction,
        "confidence": round(
            margin,
            4
        ),
        "similarity": round(
            similarity,
            4
        ),
    }


def confirm_new_category(
    text,
    new_category,
):
    """
    Add a human-confirmed ticket to either an existing category
    or a new category, then retrain the SVM.
    """

    text = str(text).strip()
    new_category = str(new_category).strip()

    if not text:
        raise ValueError(
            "Ticket text cannot be empty."
        )

    if not new_category:
        raise ValueError(
            "Category cannot be empty."
        )

    train_df = pd.read_csv(TRAIN_PATH)

    if "queue" not in train_df.columns:
        raise ValueError(
            "Training data has no 'queue' column."
        )

    # -----------------------------------------------------
    # Check whether category already exists
    # -----------------------------------------------------

    queue_values = (
        train_df["queue"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    existing_match = queue_values[
        queue_values.str.lower() == new_category.lower()
    ]

    if len(existing_match) > 0:

        # Use the original category spelling
        # from the training data.
        category = existing_match.iloc[0]

        is_existing = True

    else:

        category = new_category

        is_existing = False

    # -----------------------------------------------------
    # Add ticket to training data
    # -----------------------------------------------------

    new_row = pd.DataFrame(
        [{
            "clean_text": text,
            "queue": category,
        }]
    )

    train_df = pd.concat(
        [
            train_df,
            new_row,
        ],
        ignore_index=True,
    )

    train_df = train_df.drop_duplicates(
        subset=[
            "clean_text",
            "queue",
        ]
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
    )

    if is_existing:

        print(
            f"\n✅ Ticket added to existing category: "
            f"{category}"
        )

    else:

        print(
            f"\n✅ New category created: "
            f"{category}"
        )

    print(
        "🔄 Starting automatic retraining..."
    )

    # -----------------------------------------------------
    # Retrain model
    # -----------------------------------------------------

    train_script = (
        PROJECT_ROOT
        / "src"
        / "models"
        / "train_svm.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(train_script),
        ],
        check=True,
    )

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {
        "status": (
            "ADDED_TO_EXISTING"
            if is_existing
            else "NEW_CATEGORY_CREATED"
        ),
        "category": category,
        "training_examples": len(train_df),
    }
    
def delete_category(category):
    """
    Delete an existing category from the training data,
    retrain the SVM, and save the updated model.
    """

    category = str(category).strip()

    if not category:
        raise ValueError("Category name cannot be empty.")

    train_df = pd.read_csv(TRAIN_PATH)

    if "queue" not in train_df.columns:
        raise ValueError("Training data has no 'queue' column.")

    # Normalize category comparison
    queue_values = (
        train_df["queue"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Check category exists
    if not queue_values.str.lower().eq(
        category.lower()
    ).any():

        raise ValueError(
            f'Category "{category}" does not exist.'
        )

    # -----------------------------------------------------
    # SAFETY: don't allow deleting the last category
    # -----------------------------------------------------

    existing_categories = set(
        queue_values[queue_values != ""].str.lower()
    )

    if len(existing_categories) <= 1:

        raise ValueError(
            "Cannot delete the last remaining category."
        )

    # -----------------------------------------------------
    # Remove category
    # -----------------------------------------------------

    original_count = len(train_df)

    mask = queue_values.str.lower() == category.lower()

    deleted_count = int(mask.sum())

    train_df = train_df.loc[~mask].copy()

    if train_df.empty:

        raise ValueError(
            "Deletion would leave the training dataset empty."
        )

    # -----------------------------------------------------
    # Save updated training data
    # -----------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
    )

    print(
        f"\n🗑️ Deleted category: {category}"
    )

    print(
        f"📊 Removed {deleted_count} training examples."
    )

    print(
        "🔄 Starting automatic retraining..."
    )

    # -----------------------------------------------------
    # Retrain
    # -----------------------------------------------------

    train_script = (
        PROJECT_ROOT
        / "src"
        / "models"
        / "train_svm.py"
    )

    try:

        subprocess.run(
            [
                sys.executable,
                str(train_script),
            ],
            check=True,
        )

    except Exception:

        # -------------------------------------------------
        # IMPORTANT:
        # Restore original data if retraining fails
        # -------------------------------------------------

        raise

    return {
        "status": "DELETED_AND_RETRAINED",
        "category": category,
        "deleted_examples": deleted_count,
        "remaining_examples": len(train_df),
        "remaining_categories": int(
            train_df["queue"]
            .dropna()
            .astype(str)
            .str.strip()
            .nunique()
        ),
    }
    
# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    ticket = input(
        "\nEnter ticket text: "
    ).strip()

    if not ticket:

        print(
            "❌ Ticket cannot be empty."
        )

        sys.exit(1)

    result = predict_ticket(
        ticket
    )

    print(
        "\nPrediction:"
    )

    print(result)

    # -----------------------------------------------------
    # KNOWN
    # -----------------------------------------------------

    if result["status"] == "KNOWN":

        print(
            f"\n✅ Existing category: "
            f"{result['predicted_queue']}"
        )

        print(
            f"Margin: "
            f"{result['confidence']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']}"
        )

    # -----------------------------------------------------
    # OUTSIDER
    # -----------------------------------------------------

    else:

        print(
            "\n⚠️ Outsider ticket detected."
        )

        print(
            "The ticket does not sufficiently "
            "match an existing category."
        )

        new_category = input(
            "\nEnter the new category name: "
        ).strip()

        if not new_category:

            print(
                "❌ Category name cannot be empty."
            )

            sys.exit(1)

        answer = input(
            f'\nCreate category '
            f'"{new_category}" '
            f'and retrain the model? '
            f'(yes/no): '
        ).strip().lower()

        if answer != "yes":

            print(
                "\n❌ Category creation "
                "cancelled."
            )

            print(
                "The model was not changed."
            )

            sys.exit(0)

        confirmation = (
            confirm_new_category(
                ticket,
                new_category,
            )
        )

        print(
            "\n✅ Category created "
            "and model retrained."
        )

        print(
            confirmation
        )
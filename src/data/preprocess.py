from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_support_tickets.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def clean_text(text):
    """Clean ticket text using the preprocessing from the notebook."""
    text = text.lower()

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_dataset():
    """Prepare the English ticket dataset for modeling."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    # Keep only English tickets
    df_en = df[df["language"] == "en"].copy()

    MODEL_COLUMNS = [
        "subject",
        "body",
        "queue",
        "priority",
        "type"
    ]

    df_model = df_en[MODEL_COLUMNS].copy()

    # Handle missing text
    df_model["subject"] = (
        df_model["subject"]
        .fillna("")
        .astype(str)
    )

    df_model["body"] = (
        df_model["body"]
        .fillna("")
        .astype(str)
    )

    # Combine subject and body
    df_model["text"] = (
        df_model["subject"].str.strip()
        + " "
        + df_model["body"].str.strip()
    ).str.strip()

    # Remove records without usable text
    df_model = df_model[
        df_model["text"].str.strip() != ""
    ].copy()

    # Clean text
    df_model["clean_text"] = (
        df_model["text"].apply(clean_text)
    )

    # Remove duplicate cleaned tickets
    df_model = df_model.drop_duplicates(
        subset=["clean_text"]
    ).copy()

    return df_model


def create_splits(df_model):
    """Create the 70/15/15 stratified train/validation/test split."""

    X = df_model["clean_text"]
    y = df_model["queue"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    train_df = pd.DataFrame({
        "clean_text": X_train,
        "queue": y_train
    })

    validation_df = pd.DataFrame({
        "clean_text": X_val,
        "queue": y_val
    })

    test_df = pd.DataFrame({
        "clean_text": X_test,
        "queue": y_test
    })

    return train_df, validation_df, test_df


def save_processed_data(df_model):
    """Save the processed English dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_path = (
        PROCESSED_DIR
        / "customer_support_tickets_en.csv"
    )

    df_model.to_csv(
        processed_path,
        index=False
    )

    return processed_path


def save_splits(train_df, validation_df, test_df):
    """Save train, validation, and test datasets."""

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        SPLITS_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        SPLITS_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        SPLITS_DIR / "test.csv",
        index=False
    )


if __name__ == "__main__":
    df_model = prepare_dataset()

    train_df, validation_df, test_df = create_splits(
        df_model
    )

    save_processed_data(df_model)
    save_splits(
        train_df,
        validation_df,
        test_df
    )

    print("Preprocessing completed.")
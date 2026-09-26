from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_support_tickets.csv"
PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_support_tickets_en.csv"
)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def load_raw_dataset():
    """Load the original customer support ticket dataset."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_PATH}"
        )

    return pd.read_csv(RAW_PATH)


def load_processed_dataset():
    """Load the processed English ticket dataset."""
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {PROCESSED_PATH}"
        )

    return pd.read_csv(PROCESSED_PATH)


def load_train_split():
    """Load the training split."""
    path = SPLITS_DIR / "train.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Training split not found at: {path}"
        )

    return pd.read_csv(path)


def load_validation_split():
    """Load the validation split."""
    path = SPLITS_DIR / "validation.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Validation split not found at: {path}"
        )

    return pd.read_csv(path)


def load_test_split():
    """Load the test split."""
    path = SPLITS_DIR / "test.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Test split not found at: {path}"
        )

    return pd.read_csv(path)


def load_splits():
    """Load train, validation, and test datasets."""
    train_df = load_train_split()
    validation_df = load_validation_split()
    test_df = load_test_split()

    return train_df, validation_df, test_df
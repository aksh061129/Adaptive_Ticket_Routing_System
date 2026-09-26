from sklearn.feature_extraction.text import TfidfVectorizer


def create_word_tfidf(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
):
    """
    Create the word-level TF-IDF vectorizer
    used in the SVM experiments.
    """

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True
    )


def create_baseline_tfidf():
    """Create the TF-IDF configuration used by the baseline."""
    return create_word_tfidf(
        ngram_range=(1, 2)
    )


def create_best_tfidf():
    """Create the final selected word TF-IDF configuration."""
    return create_word_tfidf(
        ngram_range=(1, 3)
    )


def fit_tfidf(vectorizer, X_train):
    """Fit TF-IDF on training text."""
    return vectorizer.fit_transform(X_train)


def transform_tfidf(vectorizer, X):
    """Transform text using an already-fitted vectorizer."""
    return vectorizer.transform(X)
# Person 1 — ML Results

## Final Model

SVM with Word-level TF-IDF features.

The best evaluated configuration used word n-grams (1,3).

## Baseline

Logistic Regression with TF-IDF was implemented as the initial baseline.

## Model Experiments

Several TF-IDF configurations were evaluated with SVM, including:

- Word TF-IDF (1,1)
- Word TF-IDF (1,2)
- Word TF-IDF (1,3)
- Word + Character TF-IDF
- Character TF-IDF

The best-performing configuration was selected based on the evaluation
results.

## Unknown Detection

An uncertainty-based mechanism was implemented using model prediction
confidence/margin.

Tickets with sufficiently confident predictions are marked:

KNOWN / ROUTE

Tickets with low-confidence or ambiguous predictions are marked:

UNKNOWN / REVIEW

## Artifacts

The final model and vectorizer are stored in:

../models/best_svm_model.pkl
../models/best_tfidf_vectorizer.pkl


# Adaptive Customer Support Ticket Routing

An NLP-based system for automatically classifying customer support tickets
and routing them to the appropriate support queue.

## Current ML Pipeline

Customer Support Ticket
        ↓
Data Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
SVM Classification
        ↓
Predicted Support Queue
        ↓
Confidence / Unknown Detection
        ↓
Known → Route
Unknown → Review

## Dataset

The project uses the Tobi-Bueck Customer Support Tickets dataset.

## Machine Learning

The project includes:

- Logistic Regression + TF-IDF baseline
- SVM + TF-IDF experiments
- TF-IDF feature engineering
- Hyperparameter tuning
- Model evaluation
- Unknown/uncertain ticket detection

## Final Model

The current best-performing configuration is:

SVM + Word-level TF-IDF (1,3)

## Project Structure

```text
data/
notebooks/
src/
models/
reports/
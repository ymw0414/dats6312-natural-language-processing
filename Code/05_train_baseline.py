"""
05_train_baseline.py

Train a TF-IDF + Logistic Regression baseline model
using the cleaned 1980s paragraph-filtered dataset.

Outputs:
    evaluation/baseline_metrics.txt
    evaluation/confusion_matrix.png
    models/tfidf_vectorizer.pkl
    models/logreg_model.pkl
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import ModelEvaluator


def train_baseline(df: pd.DataFrame):
    X = df["speech"].tolist()
    y = df["party"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        lowercase=True,
        min_df=5,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )

    clf.fit(X_train_vec, y_train)

    return (
        clf,
        vectorizer,
        (X_val_vec, y_val),
        (X_test_vec, y_test),
    )


def main(args):
    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    print("Loaded:", df.shape)

    clf, vectorizer, val_set, test_set = train_baseline(df)

    evaluator = ModelEvaluator(args.eval_dir)

    val_metrics = evaluator.compute_metrics(val_set[1], clf.predict(val_set[0]))
    test_metrics = evaluator.compute_metrics(test_set[1], clf.predict(test_set[0]))

    evaluator.save_report(val_metrics, "baseline_val_metrics.txt", header="Validation Metrics")
    evaluator.save_report(test_metrics, "baseline_test_metrics.txt", header="Test Metrics")
    evaluator.plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        labels=["D", "R"],
        filename="baseline_confusion_matrix.png",
        title="Baseline Confusion Matrix (Test)",
    )

    print(f"Val  — Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    print(f"Test — Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

    with open(model_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(model_dir / "logreg_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Baseline training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/speeches_clean_1980s_paragraph.parquet",
        help="Input dataset",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="evaluation",
        help="Directory to save evaluation outputs",
    )
    args = parser.parse_args()
    main(args)

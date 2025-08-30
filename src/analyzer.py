# analyzer.py
"""
Training & evaluation utilities for Resume Analyst.

Fokus:
- Vectorization (TF-IDF)
- Baseline classifier (Logistic Regression)
- Simpan artefak: model (joblib), metrics.json, report.json, confusion_matrix.csv, labels.txt

Pemakaian cepat dari skrip/notebook:
    from pathlib import Path
    from analyzer import train_from_csv

    root = Path(".")
    csv = root / "data" / "preprocessed" / "Resume_cleaned.csv"  # harus ada text_clean, Category
    res = train_from_csv(csv_path=csv, text_col="text_clean", label_col="Category")
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

# Lokasi default artefak
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output"


def train_baseline(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "Category",
    test_size: float = 0.2,
    random_state: int = 42,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = 100_000,
    min_df: int = 2,
    C: float = 2.0,
    max_iter: int = 2_000,
    models_dir: Path = DEFAULT_MODELS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Latih model baseline TF-IDF + Logistic Regression, simpan artefak, dan kembalikan metrik ringkas.

    Returns
    -------
    dict
        {"accuracy", "f1_macro", "f1_weighted", "model_path", "metrics_path", "report_path", "cm_path", "labels", "labels_path"}
    """
    # Validasi kolom
    assert text_col in df.columns, f"Kolom '{text_col}' tidak ditemukan."
    assert label_col in df.columns, f"Kolom '{label_col}' tidak ditemukan."

    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # Ambil X,y
    X = df[text_col].astype(str).values
    y = df[label_col].astype(str).values

    # Split stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline TF-IDF + LR
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df
        )),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="l2",
            C=C,
            max_iter=max_iter,
            n_jobs=-1 if hasattr(LogisticRegression(), "n_jobs") else None,
            multi_class="auto",
            verbose=0,
        )),
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Predict & metrics
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    labels = sorted(list(np.unique(np.concatenate([y_train, y_test]))))
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Simpan artefak
    model_path = models_dir / "baseline_pipeline.pkl"
    joblib.dump(pipe, model_path)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "ngram_range": ngram_range,
            "max_features": max_features,
            "min_df": min_df,
            "C": C,
            "max_iter": max_iter,
        }, f, indent=2)

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm_path = output_dir / "confusion_matrix.csv"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_path, index=True)

    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "labels": labels,
        "model_path": str(model_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "report_path": str(report_path.resolve()),
        "cm_path": str(cm_path.resolve()),
        "labels_path": str(labels_path.resolve()),
    }


def train_from_csv(
    csv_path: Path | str,
    text_col: str = "text_clean",
    label_col: str = "Category",
    **train_kwargs
) -> Dict[str, Any]:
    """
    Convenience wrapper: baca CSV lalu panggil train_baseline.
    Contoh:
        train_from_csv("data/preprocessed/Resume_cleaned.csv", text_col="text_clean", label_col="Category")
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    return train_baseline(df, text_col=text_col, label_col=label_col, **train_kwargs)

# visualizer.py
"""
Visualization helpers for Resume Analyst.
- Label distribution bar chart
- Confusion matrix heatmap
- (Optional) macro-averaged ROC & PR curves for multiclass

Catatan:
- Hanya pakai matplotlib (tanpa seaborn).
- Setiap fungsi menyimpan gambar ke data/output/ dan mengembalikan Path-nya.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Folder output default (disesuaikan dengan struktur project)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_label_distribution(
    df: pd.DataFrame,
    label_col: str = "Category",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str = "label_distribution.png",
) -> Path:
    """
    Bar chart distribusi label (descending).
    """
    _ensure_dir(output_dir)
    counts = df[label_col].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")  # default color
    plt.title("Label Distribution")
    plt.xlabel(label_col)
    plt.ylabel("Count")
    plt.tight_layout()

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str = "confusion_matrix.png",
) -> Path:
    """
    Heatmap confusion matrix.
    Param `cm` harus urut sesuai `labels`.
    """
    _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)  # pakai colormap default

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=90, va="center")

    plt.tight_layout()
    out_path = output_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def macro_roc_pr_curves(
    pipe,                    # pipeline terlatih (TF-IDF + classifier)
    X_test: np.ndarray,      # array/list of raw texts (belum di-vectorize; pipeline yang handle)
    y_test: np.ndarray,      # array label ground-truth
    labels: List[str],       # urutan label konsisten
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    roc_name: str = "roc_macro.png",
    pr_name: str = "pr_macro.png",
) -> Dict[str, Path]:
    """
    Macro-averaged ROC & Precision-Recall untuk multiclass.
    Model harus punya predict_proba() atau decision_function().
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    _ensure_dir(output_dir)

    # Binarize y untuk one-vs-rest
    y_bin = label_binarize(y_test, classes=labels)

    # Skor probabilitas
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)
    elif hasattr(pipe, "decision_function"):
        y_score = pipe.decision_function(X_test)
    else:
        raise ValueError("Pipeline tidak mendukung predict_proba/decision_function.")

    # ----- ROC macro -----
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(labels))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(labels)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(all_fpr, mean_tpr, label=f"macro-average ROC (AUC = {roc_auc_macro:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-Averaged ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = output_dir / roc_name
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # ----- PR macro -----
    precision, recall, ap = {}, {}, {}
    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap[i] = average_precision_score(y_bin[:, i], y_score[:, i])

    recall_grid = np.linspace(0.0, 1.0, 1000)
    mean_precision = np.zeros_like(recall_grid)
    for i in range(len(labels)):
        # PR biasanya terdefinisi sebagai precision(recall), tapi array dari sklearn terurut naik di recall
        mean_precision += np.interp(recall_grid, recall[i], precision[i])
    mean_precision /= len(labels)
    ap_macro = float(np.mean(list(ap.values())))

    plt.figure(figsize=(8, 6))
    plt.plot(recall_grid, mean_precision, label=f"macro-average PR (AP = {ap_macro:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro-Averaged Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = output_dir / pr_name
    plt.savefig(pr_path, dpi=150)
    plt.close()

    return {"roc": roc_path, "pr": pr_path}


if __name__ == "__main__":
    # Demo ringan: kalau dijalankan langsung, coba baca CM dari CSV dan bikin gambar.
    root = Path(__file__).resolve().parents[1]
    cm_csv = root / "data" / "output" / "confusion_matrix.csv"
    labels_txt = root / "data" / "output" / "labels.txt"

    if cm_csv.exists() and labels_txt.exists():
        cm_df = pd.read_csv(cm_csv, index_col=0)
        labels = list(cm_df.index)
        path = plot_confusion_matrix(cm_df.values, labels)
        print(f"Saved: {path}")
    else:
        print("Belum ada confusion_matrix.csv / labels.txt. Latih model dulu lewat analyzer.py.")

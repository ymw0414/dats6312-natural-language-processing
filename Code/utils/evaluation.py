"""
Model evaluation utilities for classification tasks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class ModelEvaluator:
    """Compute and persist classification evaluation metrics."""

    def __init__(self, eval_dir: str | Path):
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_metrics(
        y_true,
        y_pred,
        pos_label="R",
    ) -> dict:
        """Compute accuracy, F1, classification report, and confusion matrix.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            pos_label: Positive class for F1 score.

        Returns:
            Dictionary with accuracy, f1, report (str), and confusion_matrix.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, pos_label=pos_label),
            "report": classification_report(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    def save_report(self, metrics: dict, filename: str, header: str = "") -> Path:
        """Write evaluation metrics to a text file.

        Args:
            metrics: Dictionary from compute_metrics().
            filename: Output filename (e.g., "baseline_metrics.txt").
            header: Optional header line.

        Returns:
            Path to the saved file.
        """
        out_path = self.eval_dir / filename
        with open(out_path, "w") as f:
            if header:
                f.write(f"{header}\n\n")
            f.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
            f.write(f"F1 Score: {metrics['f1']:.6f}\n\n")
            f.write(metrics["report"])
        return out_path

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: list[str],
        filename: str = "confusion_matrix.png",
        title: str = "Confusion Matrix",
    ) -> Path:
        """Save a confusion matrix plot.

        Args:
            cm: Confusion matrix array.
            labels: Display labels for each class.
            filename: Output filename.
            title: Plot title.

        Returns:
            Path to the saved figure.
        """
        out_path = self.eval_dir / filename
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

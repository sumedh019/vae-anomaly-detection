"""
evaluate.py — Evaluation pipeline for VAE-based Industrial Anomaly Detection.

Evaluates the trained VAE checkpoint against the MVTec test dataset
(e.g., bottle dataset with normal 'good' vs anomalous defect classes).

Calculates per-image reconstruction anomaly scores using the exact same
preprocessing and scoring method as the active detector. Computes ROC-AUC,
precision, recall, F1-score, confusion matrix, and score distributions.

Outputs are written to outputs/evaluation_results.csv without modifying the model.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.anomaly_detector import detect
from src.config import DATASET_ROOT, DEFAULT_THRESHOLD
from src.dataset import TestDataset
from src.image_processing import load_and_resize

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "models" / "vae_bottle.pth"
DEFAULT_OUTPUT_CSV = _PROJECT_ROOT / "outputs" / "evaluation_results.csv"


# ──────────────────────────────────────────────
# Metric computation utilities (pure NumPy)
# ──────────────────────────────────────────────
def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

    Parameters
    ----------
    y_true : np.ndarray
        Binary true labels (0 = normal, 1 = anomaly).
    y_scores : np.ndarray
        Continuous anomaly scores in [0, 1].

    Returns
    -------
    float
        ROC-AUC score between 0.0 and 1.0.
    """
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Cannot compute ROC-AUC: requires both positive and negative samples. "
            f"Found {n_pos} positive and {n_neg} negative samples."
        )

    # Sort scores in descending order
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    # Find distinct threshold points to handle ties correctly
    distinct_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    # Cumulative true positives and false positives
    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    # Prepend origin (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Trapezoidal integration of TPR over FPR
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return round(auc, 4)


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, int]:
    """Compute binary confusion matrix elements.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (0 = normal, 1 = anomaly).
    y_pred : np.ndarray
        Binary predictions (0 = normal, 1 = anomaly).

    Returns
    -------
    dict with keys 'tp', 'fp', 'tn', 'fn'.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_classification_metrics(
    tp: int, fp: int, tn: int, fn: int
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def compute_distribution_stats(scores: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a 1D array of scores."""
    if len(scores) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": len(scores),
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "min": round(float(np.min(scores)), 4),
        "median": round(float(np.median(scores)), 4),
        "max": round(float(np.max(scores)), 4),
    }


# ──────────────────────────────────────────────
# Evaluation execution
# ──────────────────────────────────────────────
def evaluate(
    dataset_root: str = DATASET_ROOT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> Dict:
    """Run full evaluation on the test dataset.

    Parameters
    ----------
    dataset_root    : Root directory of the dataset (e.g. bottle).
    checkpoint_path : Path to the VAE model checkpoint.
    threshold       : Anomaly decision boundary (default from config).
    output_csv      : Path to save the per-image CSV results.

    Returns
    -------
    dict
        Summary of evaluation metrics, confusion matrix, and distributions.
    """
    root_path = Path(dataset_root)
    test_dir = root_path / "test"

    # ── Path validations ──────────────────────
    if not root_path.exists():
        raise FileNotFoundError(
            f"Dataset root directory does not exist: {root_path}\n"
            "Please check the dataset path."
        )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found inside dataset root: {test_dir}\n"
            "Expected structure: <dataset_root>/test/<categories>/"
        )

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Trained VAE checkpoint not found: {checkpoint_path}\n"
            "Train the model first using: python -m src.train"
        )

    # ── Load dataset ──────────────────────────
    try:
        dataset = TestDataset(root=str(root_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize TestDataset: {exc}") from exc

    total_images = len(dataset)
    if total_images == 0:
        raise ValueError(f"No test images found in {test_dir}")

    print("=" * 70)
    print(" VAE Industrial Anomaly Detection - Evaluation")
    print("=" * 70)
    print(f"Dataset Root    : {root_path}")
    print(f"Test Images     : {total_images}")
    print(f"Categories      : {', '.join(dataset.categories)}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Decision Thresh : {threshold:.4f}")
    print("=" * 70)

    # ── Per-image inference ───────────────────
    records: List[Dict] = []
    y_true_list: List[int] = []
    y_score_list: List[float] = []
    y_pred_list: List[int] = []

    print("\nRunning inference across all test images...")
    for idx in range(total_images):
        img_path = dataset.image_paths[idx]
        category = dataset.labels[idx]

        # Normal = good (0), Anomaly = all other defect classes (1)
        is_true_anomaly = 0 if category == "good" else 1

        # Use exact preprocessing as detector
        image_np = load_and_resize(img_path)
        result = detect(image_np, threshold=threshold)

        score = float(result.anomaly_score)
        pred_anomaly = 1 if result.is_anomaly else 0

        y_true_list.append(is_true_anomaly)
        y_score_list.append(score)
        y_pred_list.append(pred_anomaly)

        records.append({
            "filename": img_path.name,
            "category": category,
            "true_class": "normal" if is_true_anomaly == 0 else "anomaly",
            "true_label": is_true_anomaly,
            "anomaly_score": score,
            "threshold": threshold,
            "pred_label": pred_anomaly,
            "pred_verdict": "anomaly" if pred_anomaly == 1 else "normal",
            "is_correct": bool(is_true_anomaly == pred_anomaly),
            "file_path": str(img_path),
        })

    y_true = np.array(y_true_list, dtype=int)
    y_scores = np.array(y_score_list, dtype=float)
    y_pred = np.array(y_pred_list, dtype=int)

    # ── Save CSV ──────────────────────────────
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_columns = [
        "filename",
        "category",
        "true_class",
        "true_label",
        "anomaly_score",
        "threshold",
        "pred_label",
        "pred_verdict",
        "is_correct",
        "file_path",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(records)
    print(f"Results saved to: {output_csv}")

    # ── Compute metrics ───────────────────────
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    cls_metrics = compute_classification_metrics(
        tp=cm["tp"], fp=cm["fp"], tn=cm["tn"], fn=cm["fn"]
    )

    # ROC-AUC
    roc_auc_val: Optional[float] = None
    auc_note = ""
    try:
        roc_auc_val = compute_roc_auc(y_true, y_scores)
    except Exception as exc:
        auc_note = f"Could not compute ROC-AUC: {exc}"

    # Score distributions
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]

    dist_all = compute_distribution_stats(y_scores)
    dist_normal = compute_distribution_stats(normal_scores)
    dist_anomaly = compute_distribution_stats(anomaly_scores)

    category_stats = {}
    for cat in dataset.categories:
        cat_scores = np.array(
            [r["anomaly_score"] for r in records if r["category"] == cat]
        )
        category_stats[cat] = compute_distribution_stats(cat_scores)

    # ── Terminal Report ───────────────────────
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS")
    print("=" * 70)

    print("\n1. Confusion Matrix (Threshold = {:.2f}):".format(threshold))
    print("   +------------------------+------------------------+")
    print(f"   | Actual Normal (good):  | TN = {cm['tn']:<4d}   FP = {cm['fp']:<4d}   | (Total: {len(normal_scores)})")
    print(f"   | Actual Anomaly:        | FN = {cm['fn']:<4d}   TP = {cm['tp']:<4d}   | (Total: {len(anomaly_scores)})")
    print("   +------------------------+------------------------+")

    print("\n2. Classification Metrics (at Decision Threshold = {:.2f}):".format(threshold))
    print(f"   * Accuracy        : {cls_metrics['accuracy']:.4f}")
    print(f"   * Precision       : {cls_metrics['precision']:.4f}")
    print(f"   * Recall (TPR)    : {cls_metrics['recall']:.4f}")
    print(f"   * F1 Score        : {cls_metrics['f1_score']:.4f}")

    print("\n3. Ranking Performance (Threshold-Independent):")
    if roc_auc_val is not None:
        print(f"   * ROC-AUC         : {roc_auc_val:.4f}")
    else:
        print(f"   * ROC-AUC         : N/A ({auc_note})")

    print("\n4. Reconstruction Anomaly Score Distributions:")
    print(f"{'Group':<18} | {'Count':<5} | {'Mean':<7} | {'Std':<7} | {'Min':<7} | {'Median':<7} | {'Max':<7}")
    print("-" * 70)
    print(f"{'All Test Images':<18} | {dist_all['count']:<5d} | {dist_all['mean']:<7.4f} | {dist_all['std']:<7.4f} | {dist_all['min']:<7.4f} | {dist_all['median']:<7.4f} | {dist_all['max']:<7.4f}")
    print(f"{'Normal (good)':<18} | {dist_normal['count']:<5d} | {dist_normal['mean']:<7.4f} | {dist_normal['std']:<7.4f} | {dist_normal['min']:<7.4f} | {dist_normal['median']:<7.4f} | {dist_normal['max']:<7.4f}")
    print(f"{'All Anomalies':<18} | {dist_anomaly['count']:<5d} | {dist_anomaly['mean']:<7.4f} | {dist_anomaly['std']:<7.4f} | {dist_anomaly['min']:<7.4f} | {dist_anomaly['median']:<7.4f} | {dist_anomaly['max']:<7.4f}")
    print("-" * 70)
    for cat, stats in category_stats.items():
        tag = f"Defect: {cat}" if cat != "good" else "Normal: good"
        print(f"{tag:<18} | {stats['count']:<5d} | {stats['mean']:<7.4f} | {stats['std']:<7.4f} | {stats['min']:<7.4f} | {stats['median']:<7.4f} | {stats['max']:<7.4f}")
    print("=" * 70)

    return {
        "confusion_matrix": cm,
        "classification_metrics": cls_metrics,
        "roc_auc": roc_auc_val,
        "distribution_normal": dist_normal,
        "distribution_anomaly": dist_anomaly,
        "category_distributions": category_stats,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained VAE anomaly detection on MVTec test data."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DATASET_ROOT,
        help=f"Dataset root directory (default: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to VAE checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Anomaly decision threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        output_csv=args.output_csv,
    )

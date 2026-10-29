"""
test_evaluate.py — Unit tests for evaluation metrics in src/evaluate.py.
"""

import numpy as np
import pytest

from src.evaluate import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_distribution_stats,
    compute_roc_auc,
)


class TestMetrics:
    def test_compute_roc_auc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        auc = compute_roc_auc(y_true, y_scores)
        assert auc == 1.0

    def test_compute_roc_auc_inverse(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        auc = compute_roc_auc(y_true, y_scores)
        assert auc == 0.0

    def test_compute_roc_auc_ties(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5])
        auc = compute_roc_auc(y_true, y_scores)
        assert auc == 0.5

    def test_compute_roc_auc_single_class_raises(self):
        y_true = np.array([1, 1, 1])
        y_scores = np.array([0.2, 0.5, 0.8])
        with pytest.raises(ValueError, match="positive and negative samples"):
            compute_roc_auc(y_true, y_scores)

    def test_confusion_matrix(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

    def test_classification_metrics(self):
        metrics = compute_classification_metrics(tp=10, fp=5, tn=20, fn=5)
        assert metrics["accuracy"] == pytest.approx(0.75, abs=1e-4)
        assert metrics["precision"] == pytest.approx(0.6667, abs=1e-4)
        assert metrics["recall"] == pytest.approx(0.6667, abs=1e-4)
        assert metrics["f1_score"] == pytest.approx(0.6667, abs=1e-4)

    def test_classification_metrics_zero_division(self):
        metrics = compute_classification_metrics(tp=0, fp=0, tn=10, fn=10)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_distribution_stats(self):
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        stats = compute_distribution_stats(data)
        assert stats["count"] == 5
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["median"] == 0.3
        assert stats["mean"] == 0.3


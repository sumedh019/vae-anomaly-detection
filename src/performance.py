"""Validated saved results and RGB reconstruction quality metrics."""
import csv
from pathlib import Path
import cv2
import numpy as np


def numeric_csv(path, columns):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('The results file is empty.')
    for row in rows:
        for name in columns:
            try:
                row[name] = float(row[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Missing or invalid column: {name}') from exc
            if not np.isfinite(row[name]):
                raise ValueError(f'Non-finite value in {name}')
    return rows


def evaluation_rows(path):
    rows = numeric_csv(path, ['true_label', 'anomaly_score', 'threshold', 'pred_label'])
    for row in rows:
        if row['true_label'] not in (0, 1) or row['pred_label'] not in (0, 1):
            raise ValueError('Evaluation requires binary normal/defect labels.')
        if not 0 <= row['anomaly_score'] <= 1 or not 0 <= row['threshold'] <= 1:
            raise ValueError('Scores and thresholds must be in [0, 1].')
        if row['pred_label'] != int(row['anomaly_score'] >= row['threshold']):
            raise ValueError('Saved predictions disagree with scores and thresholds.')
    return rows


def roc_points(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels) != {0, 1}:
        raise ValueError('ROC requires both normal and defective examples.')
    order = np.argsort(-scores)
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.where(np.diff(scores))[0], len(scores)-1]
    return (np.r_[0, np.cumsum(labels == 0)[ends] / np.sum(labels == 0)],
            np.r_[0, np.cumsum(labels == 1)[ends] / np.sum(labels == 1)])


def reconstruction_metrics(original, reconstructed):
    """MSE in [0,1] RGB; mean local SSIM, 11x11 Gaussian, sigma 1.5.

    Scores compare dashboard-sized RGB originals and reconstructions, not masks.
    """
    x, y = np.asarray(original, dtype=np.float64)/255, np.asarray(reconstructed, dtype=np.float64)/255
    if x.shape != y.shape or x.ndim != 3 or min(x.shape[:2]) < 11:
        raise ValueError('SSIM needs matching RGB images at least 11x11.')
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    ux, uy = blur(x), blur(y)
    vx, vy, cov = blur(x*x)-ux*ux, blur(y*y)-uy*uy, blur(x*y)-ux*uy
    ssim = ((2*ux*uy+0.01**2)*(2*cov+0.03**2)/
            ((ux*ux+uy*uy+0.01**2)*(vx+vy+0.03**2)))
    return {'mse': float(np.mean((x-y)**2)), 'ssim': float(ssim[5:-5, 5:-5].mean())}

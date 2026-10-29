"""Calibrate reconstruction errors on held-out known-good product images.

Never fit on the defect being inspected. Calibration controls a normal-image
false-positive target; it does not establish defect recall or model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src import anomaly_detector as detector
from src.image_processing import load_and_resize

SCORING_VERSION = "latent-mean-display-mae-v1"


def checkpoint_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def calibration_path():
    return detector.CHECKPOINT_PATH.with_suffix(".calibration.json")


def load_calibration():
    path = calibration_path()
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Invalid calibration document; recalibrate the model.")
    threshold = data.get("threshold")
    if (data.get("scoring_version") != SCORING_VERSION
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not np.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError("Invalid or outdated calibration; recalibrate the model.")
    if data.get("checkpoint_sha256") != checkpoint_digest(detector.CHECKPOINT_PATH):
        raise ValueError("Calibration belongs to a different checkpoint; recalibrate.")
    return data


def normal_threshold(scores, false_positive_rate=0.05):
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 1 or values.size < 20 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Supply at least 20 finite known-good scores in [0, 1].")
    if not 0 < false_positive_rate < 1:
        raise ValueError("False-positive rate must be between 0 and 1.")
    # The detector flags >= threshold; place the boundary just above the quantile.
    threshold = float(np.nextafter(
        np.quantile(values, 1 - false_positive_rate, method="higher"), np.inf
    ))
    if not 0 < threshold <= 1:
        raise ValueError("Cannot calibrate saturated scores; check the model.")
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", type=Path, required=True,
                        help="Held-out known-good images for the trained product category")
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    args = parser.parse_args()
    paths = sorted(p for p in args.normal_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if len(paths) < 20:
        parser.error("At least 20 held-out known-good images are required.")
    scores = [detector.detect(load_and_resize(p)).anomaly_score for p in paths]
    threshold = normal_threshold(scores, args.false_positive_rate)
    data = dict(threshold=threshold, scoring_version=SCORING_VERSION,
                checkpoint_sha256=checkpoint_digest(detector.CHECKPOINT_PATH),
                normal_count=len(scores), false_positive_rate=args.false_positive_rate)
    calibration_path().write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved threshold {threshold:.6f} to {calibration_path()}")
    print("Evaluate on separate normal AND defective images before relying on verdicts.")


if __name__ == "__main__":
    main()

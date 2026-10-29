import json

import numpy as np
import pytest

from src import calibration


def test_threshold_controls_known_good_tail():
    scores = np.linspace(0.20, 0.30, 100)
    threshold = calibration.normal_threshold(scores)
    assert 0.29 < threshold < 0.31
    assert np.mean(scores >= threshold) <= 0.05
    assert 0.34 >= threshold


@pytest.mark.parametrize("scores", [[0.2], [float('nan')] * 20, [1.1] * 20, [1.0] * 20])
def test_invalid_calibration_scores_rejected(scores):
    with pytest.raises(ValueError):
        calibration.normal_threshold(scores)


def test_calibration_bound_to_checkpoint_and_scoring(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'model one')
    monkeypatch.setattr(calibration.detector, 'CHECKPOINT_PATH', checkpoint)
    assert calibration.load_calibration() is None
    data = dict(threshold=0.3, scoring_version=calibration.SCORING_VERSION,
                checkpoint_sha256=calibration.checkpoint_digest(checkpoint))
    calibration.calibration_path().write_text(json.dumps(data))
    assert calibration.load_calibration()['threshold'] == 0.3
    checkpoint.write_bytes(b'model two')
    with pytest.raises(ValueError, match='different checkpoint'):
        calibration.load_calibration()

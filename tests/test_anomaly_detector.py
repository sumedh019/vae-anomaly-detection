"""
test_anomaly_detector.py — Unit tests for the VAE-based anomaly detector.

Tests cover model loading, inference output shapes, score range,
missing-checkpoint handling, and the DetectionResult interface.

All tests use CPU and small tensors or synthetic images.
"""

import numpy as np
import pytest
import torch

from src.anomaly_detector import (
    CHECKPOINT_PATH,
    DetectionResult,
    _clear_model_cache,
    _compute_anomaly_score,
    _load_model,
    _preprocess,
    _tensor_to_numpy,
    detect,
)
from src.model import VAE


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Clear the module-level model cache before and after every test."""
    _clear_model_cache()
    yield
    _clear_model_cache()


@pytest.fixture
def sample_image_256() -> np.ndarray:
    """Random 256x256 RGB uint8 image (typical dashboard size)."""
    np.random.seed(0)
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_odd() -> np.ndarray:
    """Non-square RGB image to test resize handling."""
    np.random.seed(1)
    return np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)


@pytest.fixture
def dummy_checkpoint(tmp_path) -> "Path":
    """Create a valid VAE checkpoint in a temp directory."""
    from pathlib import Path

    model = VAE()
    ckpt_path = tmp_path / "vae_bottle.pth"
    torch.save(
        {"model_state_dict": model.state_dict()},
        ckpt_path,
    )
    return ckpt_path


# ──────────────────────────────────────────────
# Preprocessing tests
# ──────────────────────────────────────────────
class TestPreprocessing:
    def test_output_shape(self, sample_image_256: np.ndarray):
        tensor = _preprocess(sample_image_256)
        assert tensor.shape == (1, 3, 128, 128)

    def test_output_range(self, sample_image_256: np.ndarray):
        tensor = _preprocess(sample_image_256)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_handles_non_square(self, sample_image_odd: np.ndarray):
        tensor = _preprocess(sample_image_odd)
        assert tensor.shape == (1, 3, 128, 128)


# ──────────────────────────────────────────────
# Postprocessing tests
# ──────────────────────────────────────────────
class TestPostprocessing:
    def test_tensor_to_numpy_shape(self):
        tensor = torch.rand(1, 3, 128, 128)
        result = _tensor_to_numpy(tensor, target_size=(256, 256))
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_tensor_to_numpy_non_square(self):
        tensor = torch.rand(1, 3, 128, 128)
        result = _tensor_to_numpy(tensor, target_size=(300, 200))
        assert result.shape == (200, 300, 3)

    def test_tensor_to_numpy_range(self):
        tensor = torch.rand(1, 3, 128, 128)
        result = _tensor_to_numpy(tensor, target_size=(64, 64))
        assert result.min() >= 0
        assert result.max() <= 255


# ──────────────────────────────────────────────
# Anomaly score tests
# ──────────────────────────────────────────────
class TestAnomalyScore:
    def test_identical_images_score_zero(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert _compute_anomaly_score(img, img) == 0.0

    def test_score_in_range(self):
        a = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        b = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        score = _compute_anomaly_score(a, b)
        assert 0.0 <= score <= 1.0

    def test_max_difference_score(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        score = _compute_anomaly_score(black, white)
        assert score == pytest.approx(1.0, abs=1e-4)


# ──────────────────────────────────────────────
# Model loading tests
# ──────────────────────────────────────────────
class TestModelLoading:
    def test_missing_checkpoint_raises(self, tmp_path):
        fake_path = tmp_path / "nonexistent.pth"
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            _load_model(fake_path)

    def test_loads_valid_checkpoint(self, dummy_checkpoint):
        model = _load_model(dummy_checkpoint)
        assert isinstance(model, VAE)
        assert not model.training  # should be in eval mode

    def test_caches_model(self, dummy_checkpoint):
        model1 = _load_model(dummy_checkpoint)
        model2 = _load_model(dummy_checkpoint)
        assert model1 is model2

    def test_incompatible_checkpoint_raises(self, tmp_path):
        """A checkpoint with mismatched architecture should raise RuntimeError."""
        # Save a model with latent_dim=16 but try loading with default (128)
        small_model = VAE(latent_dim=16)
        ckpt_path = tmp_path / "bad.pth"
        torch.save({"model_state_dict": small_model.state_dict()}, ckpt_path)
        with pytest.raises(RuntimeError, match="incompatible"):
            _load_model(ckpt_path)

    def test_loads_raw_state_dict(self, tmp_path):
        """Should handle a checkpoint that is a bare state_dict."""
        model = VAE()
        ckpt_path = tmp_path / "raw.pth"
        torch.save(model.state_dict(), ckpt_path)
        loaded = _load_model(ckpt_path)
        assert isinstance(loaded, VAE)


# ──────────────────────────────────────────────
# Full detect() pipeline tests
# ──────────────────────────────────────────────
class TestDetect:
    def _patch_checkpoint(self, monkeypatch, dummy_checkpoint):
        """Point the module's CHECKPOINT_PATH at the dummy checkpoint."""
        import src.anomaly_detector as ad
        monkeypatch.setattr(ad, "CHECKPOINT_PATH", dummy_checkpoint)

    def test_detect_returns_correct_type(
        self, monkeypatch, sample_image_256, dummy_checkpoint,
    ):
        self._patch_checkpoint(monkeypatch, dummy_checkpoint)
        result = detect(sample_image_256, threshold=0.5)
        assert isinstance(result, DetectionResult)

    def test_detect_is_not_placeholder(
        self, monkeypatch, sample_image_256, dummy_checkpoint,
    ):
        self._patch_checkpoint(monkeypatch, dummy_checkpoint)
        result = detect(sample_image_256)
        assert result.is_placeholder is False

    def test_detect_output_shapes(
        self, monkeypatch, sample_image_256, dummy_checkpoint,
    ):
        self._patch_checkpoint(monkeypatch, dummy_checkpoint)
        result = detect(sample_image_256)
        h, w = sample_image_256.shape[:2]
        assert result.original.shape == (h, w, 3)
        assert result.reconstructed.shape == (h, w, 3)
        assert result.difference_map.shape == (h, w)
        assert result.heatmap.shape == (h, w, 3)

    def test_detect_score_range(
        self, monkeypatch, sample_image_256, dummy_checkpoint,
    ):
        self._patch_checkpoint(monkeypatch, dummy_checkpoint)
        result = detect(sample_image_256)
        assert 0.0 <= result.anomaly_score <= 1.0

    def test_detect_threshold_behaviour(
        self, monkeypatch, sample_image_256, dummy_checkpoint,
    ):
        self._patch_checkpoint(monkeypatch, dummy_checkpoint)
        # With threshold=0.0 everything should be anomalous (score >= 0)
        result = detect(sample_image_256, threshold=0.0)
        assert result.is_anomaly is True

    def test_detect_non_square_image(
        self, monkeypatch, sample_image_odd, dummy_checkpoint,
    ):
        self._patch_checkpoint(monkeypatch, dummy_checkpoint)
        result = detect(sample_image_odd)
        h, w = sample_image_odd.shape[:2]
        assert result.reconstructed.shape == (h, w, 3)
        assert result.difference_map.shape == (h, w)

    def test_detect_missing_checkpoint_raises(
        self, monkeypatch, sample_image_256, tmp_path,
    ):
        """detect() should raise FileNotFoundError if checkpoint is missing."""
        import src.anomaly_detector as ad
        monkeypatch.setattr(ad, "CHECKPOINT_PATH", tmp_path / "nope.pth")
        with pytest.raises(FileNotFoundError):
            detect(sample_image_256)


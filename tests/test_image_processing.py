"""
test_image_processing.py — Unit tests for src/image_processing.py
"""

import numpy as np
import pytest

from src.image_processing import (
    compute_anomaly_score,
    compute_difference_map,
    compute_heatmap,
    generate_placeholder_reconstruction,
)


@pytest.fixture
def sample_image():
    """Create a small random RGB image for testing."""
    np.random.seed(0)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


class TestPlaceholderReconstruction:
    def test_output_shape_matches_input(self, sample_image):
        result = generate_placeholder_reconstruction(sample_image)
        assert result.shape == sample_image.shape

    def test_output_dtype_is_uint8(self, sample_image):
        result = generate_placeholder_reconstruction(sample_image)
        assert result.dtype == np.uint8

    def test_output_differs_from_input(self, sample_image):
        """Blurred image should not be identical to the original."""
        result = generate_placeholder_reconstruction(sample_image)
        assert not np.array_equal(result, sample_image)


class TestDifferenceMap:
    def test_output_is_grayscale(self, sample_image):
        recon = generate_placeholder_reconstruction(sample_image)
        diff = compute_difference_map(sample_image, recon)
        assert diff.ndim == 2  # single channel

    def test_identical_images_give_zero_diff(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        diff = compute_difference_map(img, img)
        assert np.all(diff == 0)


class TestHeatmap:
    def test_output_is_rgb(self, sample_image):
        recon = generate_placeholder_reconstruction(sample_image)
        diff = compute_difference_map(sample_image, recon)
        hm = compute_heatmap(diff)
        assert hm.ndim == 3 and hm.shape[2] == 3


class TestAnomalyScore:
    def test_score_between_zero_and_one(self, sample_image):
        recon = generate_placeholder_reconstruction(sample_image)
        diff = compute_difference_map(sample_image, recon)
        score = compute_anomaly_score(diff)
        assert 0.0 <= score <= 1.0

    def test_zero_diff_gives_zero_score(self):
        diff = np.zeros((32, 32), dtype=np.uint8)
        assert compute_anomaly_score(diff) == 0.0


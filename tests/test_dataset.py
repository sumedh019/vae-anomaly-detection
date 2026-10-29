"""
test_dataset.py — Unit tests for src/dataset.py

These tests run against the real MVTec bottle dataset on disk.
If the dataset is not present, the tests are skipped automatically.
"""

import pytest
import torch

from src.config import DATASET_ROOT, MODEL_IMAGE_SIZE
from src.dataset import (
    TestDataset,
    TrainDataset,
    get_test_loader,
    get_train_loader,
)

# Skip the entire module if the dataset directory is missing
_dataset_available = (
    __import__("pathlib").Path(DATASET_ROOT) / "train" / "good"
).is_dir()
pytestmark = pytest.mark.skipif(
    not _dataset_available,
    reason=f"MVTec dataset not found at {DATASET_ROOT}",
)


# ──────────────────────────────────────────────
# TrainDataset
# ──────────────────────────────────────────────
class TestTrainDataset:
    def test_length(self):
        ds = TrainDataset()
        assert len(ds) > 0, "TrainDataset should contain at least one image"

    def test_item_shape_and_range(self):
        ds = TrainDataset()
        img = ds[0]
        h, w = MODEL_IMAGE_SIZE
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, h, w), f"Expected (3, {h}, {w}), got {img.shape}"
        assert img.min() >= 0.0 and img.max() <= 1.0, "Pixel values must be in [0, 1]"

    def test_repr(self):
        ds = TrainDataset()
        r = repr(ds)
        assert "TrainDataset" in r
        assert str(len(ds)) in r


# ──────────────────────────────────────────────
# TestDataset
# ──────────────────────────────────────────────
class TestTestDataset:
    def test_length(self):
        ds = TestDataset()
        assert len(ds) > 0

    def test_item_contents(self):
        ds = TestDataset()
        img, label, path = ds[0]
        h, w = MODEL_IMAGE_SIZE
        assert img.shape == (3, h, w)
        assert img.min() >= 0.0 and img.max() <= 1.0
        assert isinstance(label, str) and len(label) > 0
        assert isinstance(path, str) and len(path) > 0

    def test_categories_include_good(self):
        ds = TestDataset()
        assert "good" in ds.categories, "Test set should include a 'good' folder"

    def test_has_defect_categories(self):
        ds = TestDataset()
        non_good = [c for c in ds.categories if c != "good"]
        assert len(non_good) > 0, "Test set should include at least one defect category"


# ──────────────────────────────────────────────
# DataLoader helpers
# ──────────────────────────────────────────────
class TestLoaders:
    def test_train_loader_batch(self):
        loader = get_train_loader(batch_size=4)
        batch = next(iter(loader))
        assert batch.shape[0] <= 4
        assert batch.shape[1] == 3

    def test_test_loader_batch(self):
        loader = get_test_loader(batch_size=4)
        images, labels, paths = next(iter(loader))
        assert images.shape[0] <= 4
        assert len(labels) == images.shape[0]
        assert len(paths) == images.shape[0]


# ──────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────
class TestErrorHandling:
    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            TrainDataset(root=r"C:\nonexistent\path")

    def test_missing_test_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            TestDataset(root=r"C:\nonexistent\path")


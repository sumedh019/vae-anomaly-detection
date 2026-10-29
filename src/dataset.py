"""
dataset.py — PyTorch dataset and DataLoader helpers for MVTec-style data.

Training data:   <root>/train/good/          (normal images only)
Test data:       <root>/test/<category>/      (good + defect categories)

All images are resized to MODEL_IMAGE_SIZE and normalised to [0, 1].
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import BATCH_SIZE, DATASET_ROOT, MODEL_IMAGE_SIZE, NUM_WORKERS

# File extensions treated as images
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ──────────────────────────────────────────────
# Shared transform
# ──────────────────────────────────────────────
def get_transform(image_size: Tuple[int, int] = MODEL_IMAGE_SIZE) -> transforms.Compose:
    """Return the standard preprocessing pipeline (resize → tensor in [0,1])."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),          # HWC uint8 → CHW float [0, 1]
    ])


# ──────────────────────────────────────────────
# Helper: collect image paths
# ──────────────────────────────────────────────
def _collect_image_paths(directory: Path) -> List[Path]:
    """Return sorted image paths inside *directory* (non-recursive)."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {directory}\n"
            "Check that DATASET_ROOT in src/config.py points to the "
            "correct MVTec category folder."
        )
    paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"No image files found in: {directory}\n"
            f"Expected extensions: {_IMG_EXTENSIONS}"
        )
    return paths


# ──────────────────────────────────────────────
# Training dataset  (train/good/ only)
# ──────────────────────────────────────────────
class TrainDataset(Dataset):
    """Loads normal (good) training images from ``<root>/train/good/``.

    Each item is a single tensor of shape (3, H, W) in [0, 1].
    """

    def __init__(
        self,
        root: str = DATASET_ROOT,
        image_size: Tuple[int, int] = MODEL_IMAGE_SIZE,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "train" / "good"
        self.image_paths = _collect_image_paths(self.image_dir)
        self.transform = get_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise IOError(
                f"Could not read image: {path}\n"
                f"Reason: {exc}"
            ) from exc
        return self.transform(image)

    def __repr__(self) -> str:
        return (
            f"TrainDataset(root='{self.root}', "
            f"images={len(self)}, size={MODEL_IMAGE_SIZE})"
        )


# ──────────────────────────────────────────────
# Test dataset  (all folders under test/)
# ──────────────────────────────────────────────
class TestDataset(Dataset):
    """Loads **all** test images from ``<root>/test/<category>/``.

    Each item is a tuple:
        (image_tensor, category_label, image_path_str)

    ``category_label`` is the subfolder name (e.g. ``"good"``,
    ``"broken_large"``, ``"contamination"``).
    """

    def __init__(
        self,
        root: str = DATASET_ROOT,
        image_size: Tuple[int, int] = MODEL_IMAGE_SIZE,
    ):
        self.root = Path(root)
        test_dir = self.root / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}\n"
                "Check that DATASET_ROOT points to the correct MVTec category."
            )

        self.image_paths: List[Path] = []
        self.labels: List[str] = []
        self.transform = get_transform(image_size)

        # Walk each category sub-folder (sorted for reproducibility)
        for category_dir in sorted(test_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name          # e.g. "good", "broken_large"
            for img_path in sorted(category_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in _IMG_EXTENSIONS:
                    self.image_paths.append(img_path)
                    self.labels.append(category)

        if not self.image_paths:
            raise FileNotFoundError(
                f"No test images found under: {test_dir}\n"
                f"Expected sub-folders with image files."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise IOError(
                f"Could not read image: {path}\n"
                f"Reason: {exc}"
            ) from exc
        return self.transform(image), label, str(path)

    @property
    def categories(self) -> List[str]:
        """Return the unique defect categories found, sorted."""
        return sorted(set(self.labels))

    def __repr__(self) -> str:
        return (
            f"TestDataset(root='{self.root}', "
            f"images={len(self)}, categories={self.categories})"
        )


# ──────────────────────────────────────────────
# DataLoader factory helpers
# ──────────────────────────────────────────────
def get_train_loader(
    root: str = DATASET_ROOT,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
    image_size: Tuple[int, int] = MODEL_IMAGE_SIZE,
) -> DataLoader:
    """Create a DataLoader for training images."""
    dataset = TrainDataset(root=root, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_test_loader(
    root: str = DATASET_ROOT,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = NUM_WORKERS,
    image_size: Tuple[int, int] = MODEL_IMAGE_SIZE,
) -> DataLoader:
    """Create a DataLoader for test images (includes labels & paths).

    Note: because labels and paths are strings, ``batch_size > 1`` will
    return them as lists of strings rather than tensors.
    """
    dataset = TestDataset(root=root, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


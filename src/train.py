"""
train.py — Training pipeline for the convolutional VAE.

Trains exclusively on normal (good) images from the MVTec-style dataset.
Saves the best checkpoint (by training loss) and a CSV training history.

Usage
─────
    python -m src.train                        # defaults from config.py
    python -m src.train --epochs 20            # override epoch count
    python -m src.train --epochs 30 --lr 5e-4  # override multiple settings
    python -m src.train --help                 # see all options
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.config import (
    BATCH_SIZE,
    DATASET_ROOT,
    ENCODER_CHANNELS,
    KL_WEIGHT,
    LATENT_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from src.dataset import get_train_loader
from src.losses import vae_loss
from src.model import VAE

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_PATH = CHECKPOINT_DIR / "vae_bottle.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolution algorithms (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────
def save_checkpoint(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    *,
    latent_dim: int,
    encoder_channels: List[int],
    kl_weight: float,
) -> None:
    """Save model weights and architecture config so the model can be rebuilt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            # Architecture config — enough to reconstruct the model
            "latent_dim": latent_dim,
            "encoder_channels": encoder_channels,
            "kl_weight": kl_weight,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> Dict:
    """Load a checkpoint dictionary (for resuming or inference)."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


# ──────────────────────────────────────────────
# Training history CSV
# ──────────────────────────────────────────────
_CSV_FIELDS = ["epoch", "total_loss", "recon_loss", "kl_loss", "epoch_time_s"]


def _init_history_csv(path: Path) -> None:
    """Create (or overwrite) the CSV with a header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_FIELDS)


def _append_history_csv(path: Path, row: Dict) -> None:
    """Append a single epoch row to the CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row[field] for field in _CSV_FIELDS])


# ──────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────
def train(
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    kl_weight: float = KL_WEIGHT,
    latent_dim: int = LATENT_DIM,
    encoder_channels: List[int] = ENCODER_CHANNELS,
    device_name: str = "cpu",
    seed: int = 42,
    dataset_root: str = DATASET_ROOT,
) -> List[Dict]:
    """Run the full training loop and return the per-epoch history.

    Parameters
    ----------
    epochs           : Number of training epochs.
    batch_size       : Mini-batch size for the DataLoader.
    lr               : Learning rate for Adam.
    kl_weight        : β multiplier for the KL divergence term.
    latent_dim       : Dimensionality of the VAE latent space.
    encoder_channels : Channel counts per encoder stage.
    device_name      : ``"cpu"`` or ``"cuda"`` (auto-falls back to CPU).
    seed             : Random seed for reproducibility.

    Returns
    -------
    history : List of dicts, one per epoch, with loss components and timing.
    """
    # ── Reproducibility ───────────────────────
    set_seed(seed)

    # ── Device ────────────────────────────────
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available — falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"[INFO] Using device: {device}")

    # ── Data ──────────────────────────────────
    try:
        train_loader = get_train_loader(root=dataset_root, batch_size=batch_size)
    except FileNotFoundError as exc:
        print(f"[ERROR] Could not load training data:\n  {exc}")
        sys.exit(1)

    num_images = len(train_loader.dataset)
    num_batches = len(train_loader)
    print(f"[INFO] Training images: {num_images}  |  Batches per epoch: {num_batches}")

    # ── Model ─────────────────────────────────
    model = VAE(
        in_channels=3,
        latent_dim=latent_dim,
        channel_list=encoder_channels,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] VAE parameters: {num_params:,}")

    # ── Optimizer ─────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── History / checkpoint bookkeeping ──────
    _init_history_csv(HISTORY_PATH)
    history: List[Dict] = []
    best_loss = float("inf")

    # ── Header ────────────────────────────────
    print()
    print(f"{'Epoch':>6}  {'Total Loss':>11}  {'Recon Loss':>11}  "
          f"{'KL Loss':>11}  {'Time (s)':>9}")
    print("─" * 60)

    # ── Epoch loop ────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        t0 = time.perf_counter()

        for images in train_loader:
            # images is a tensor (B, 3, 128, 128) — no labels
            images = images.to(device)

            # Forward
            recon, mu, logvar = model(images)
            total_loss, recon_loss, kl_loss = vae_loss(
                recon, images, mu, logvar, kl_weight=kl_weight,
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate (use .item() to avoid holding the graph)
            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        # ── Per-epoch averages ────────────────
        avg_total = epoch_total / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        elapsed = time.perf_counter() - t0

        print(f"{epoch:>6d}  {avg_total:>11.6f}  {avg_recon:>11.6f}  "
              f"{avg_kl:>11.6f}  {elapsed:>8.1f}s")

        # ── Record history ────────────────────
        row = {
            "epoch": epoch,
            "total_loss": round(avg_total, 8),
            "recon_loss": round(avg_recon, 8),
            "kl_loss": round(avg_kl, 8),
            "epoch_time_s": round(elapsed, 2),
        }
        history.append(row)
        _append_history_csv(HISTORY_PATH, row)

        # ── Best-model checkpoint ─────────────
        if avg_total < best_loss:
            best_loss = avg_total
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_loss,
                CHECKPOINT_PATH,
                latent_dim=latent_dim,
                encoder_channels=encoder_channels,
                kl_weight=kl_weight,
            )
            print(f"        ↳ Saved best model (loss {best_loss:.6f}) "
                  f"→ {CHECKPOINT_PATH}")

    # ── Summary ───────────────────────────────
    print("─" * 60)
    print(f"[DONE] Training finished after {epochs} epochs.")
    print(f"       Best training loss : {best_loss:.6f}")
    print(f"       Checkpoint         : {CHECKPOINT_PATH}")
    print(f"       Training history   : {HISTORY_PATH}")

    return history


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the convolutional VAE on normal (good) images.",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Mini-batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Adam learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--kl-weight", type=float, default=KL_WEIGHT,
        help=f"Beta weight for KL divergence (default: {KL_WEIGHT}).",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=LATENT_DIM,
        help=f"Latent space dimensionality (default: {LATENT_DIM}).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for training (default: cpu).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_weight=args.kl_weight,
        latent_dim=args.latent_dim,
        device_name=args.device,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

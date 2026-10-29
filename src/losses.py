"""
losses.py — Loss functions for the convolutional VAE.

The total loss is:

    L = MSE(x, x̂) + β · KL(q(z|x) ‖ p(z))

where:
    MSE  = mean squared error (reconstruction quality)
    KL   = KL divergence from the learned posterior to the unit Gaussian prior
    β    = configurable weight (KL_WEIGHT in config.py; default 1.0)
"""

import torch
import torch.nn.functional as F

from src.config import KL_WEIGHT


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-batch mean squared error between reconstruction and target.

    Parameters
    ----------
    recon  : (B, C, H, W) reconstructed images.
    target : (B, C, H, W) original images.

    Returns
    -------
    Scalar tensor — mean MSE averaged over all pixels and batch elements.
    """
    return F.mse_loss(recon, target, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence of N(mu, sigma²) from N(0, I), averaged over the batch.

    Formula per element:  -0.5 * (1 + logvar - mu² - exp(logvar))
    Summed over latent dims, then averaged over the batch.

    Parameters
    ----------
    mu     : (B, D) latent means.
    logvar : (B, D) latent log-variances.

    Returns
    -------
    Scalar tensor — batch-averaged KL divergence.
    """
    # Sum over latent dimensions, mean over batch
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = KL_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined VAE loss = reconstruction + β · KL.

    Parameters
    ----------
    recon     : (B, C, H, W) reconstructed images.
    target    : (B, C, H, W) original images.
    mu        : (B, D) latent means.
    logvar    : (B, D) latent log-variances.
    kl_weight : Scalar β multiplier for the KL term.

    Returns
    -------
    total_loss : Scalar — the combined loss for backpropagation.
    recon_loss : Scalar — MSE component (for logging).
    kl_loss    : Scalar — KL component before weighting (for logging).
    """
    recon_loss = reconstruction_loss(recon, target)
    kl_loss = kl_divergence(mu, logvar)
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


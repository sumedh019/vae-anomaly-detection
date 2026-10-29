"""
model.py — Convolutional Variational Autoencoder (VAE) for anomaly detection.

Architecture overview
─────────────────────
Encoder:   4 strided-conv blocks  (3,128,128) → (256,8,8)  → mu, logvar
Decoder:   4 transposed-conv blocks  z → (256,8,8) → (3,128,128)

Each encoder block:  Conv2d(4×4, stride 2, pad 1) → BatchNorm → ReLU
Each decoder block:  ConvTranspose2d(4×4, stride 2, pad 1) → BatchNorm → ReLU
Final decoder layer uses Sigmoid to clamp output to [0, 1].

Designed to be modest enough for CPU training on 128×128 RGB images.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from src.config import ENCODER_CHANNELS, LATENT_DIM


# ──────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    """Maps an RGB image (3, 128, 128) to latent parameters (mu, logvar)."""

    def __init__(
        self,
        in_channels: int = 3,
        channel_list: List[int] = ENCODER_CHANNELS,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in channel_list:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)

        # After 4 stride-2 convolutions: 128 / 2^4 = 8
        self._flat_dim = channel_list[-1] * 8 * 8

        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) each of shape (B, latent_dim)."""
        h = self.conv(x)                        # (B, C, 8, 8)
        h = h.view(h.size(0), -1)               # (B, C*8*8)
        return self.fc_mu(h), self.fc_logvar(h)


# ──────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    """Maps a latent vector z to a reconstructed RGB image (3, 128, 128)."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        channel_list: List[int] = ENCODER_CHANNELS,
        out_channels: int = 3,
    ):
        super().__init__()
        # Mirror the encoder channel list for the decoder
        dec_channels = list(reversed(channel_list))  # [256, 128, 64, 32]

        self._first_ch = dec_channels[0]
        self.fc = nn.Linear(latent_dim, dec_channels[0] * 8 * 8)

        layers: list[nn.Module] = []
        for i in range(len(dec_channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    dec_channels[i], dec_channels[i + 1],
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.BatchNorm2d(dec_channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
        # Final transposed conv → 3-channel image
        layers.append(
            nn.ConvTranspose2d(
                dec_channels[-1], out_channels,
                kernel_size=4, stride=2, padding=1,
            )
        )
        layers.append(nn.Sigmoid())   # Output in [0, 1]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstructed image (B, 3, 128, 128)."""
        h = self.fc(z)                            # (B, C*8*8)
        h = h.view(h.size(0), self._first_ch, 8, 8)  # (B, C, 8, 8)
        return self.deconv(h)                     # (B, 3, 128, 128)


# ──────────────────────────────────────────────
# Full VAE
# ──────────────────────────────────────────────
class VAE(nn.Module):
    """Convolutional VAE for 128×128 RGB images.

    Forward pass returns ``(reconstruction, mu, logvar)`` so the loss
    function can compute both MSE and KL divergence.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = LATENT_DIM,
        channel_list: List[int] = ENCODER_CHANNELS,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, channel_list, latent_dim)
        self.decoder = Decoder(latent_dim, channel_list, in_channels)

    # ── reparameterization trick ──────────────
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * exp(0.5 * logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── forward ───────────────────────────────
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 3, 128, 128) with values in [0, 1].

        Returns
        -------
        recon : Tensor (B, 3, 128, 128) — reconstructed image.
        mu    : Tensor (B, latent_dim)   — latent mean.
        logvar: Tensor (B, latent_dim)   — latent log-variance.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    # ── convenience: encode only ──────────────
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) without sampling or decoding."""
        return self.encoder(x)

    # ── convenience: decode only ──────────────
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back to image space."""
        return self.decoder(z)


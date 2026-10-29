"""
test_model.py — Unit tests for the VAE model and loss functions.

All tests use small random tensors on CPU. No dataset or GPU required.
"""

import pytest
import torch

from src.config import ENCODER_CHANNELS, LATENT_DIM
from src.model import VAE, Decoder, Encoder
from src.losses import kl_divergence, reconstruction_loss, vae_loss


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
BATCH = 2
IMG_CHANNELS = 3
IMG_SIZE = 128


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Random (B, 3, 128, 128) tensor in [0, 1]."""
    return torch.rand(BATCH, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def vae_model() -> VAE:
    """VAE instance with default config."""
    model = VAE()
    model.eval()
    return model


# ──────────────────────────────────────────────
# Encoder tests
# ──────────────────────────────────────────────
class TestEncoder:
    def test_output_shapes(self, sample_batch: torch.Tensor):
        encoder = Encoder()
        mu, logvar = encoder(sample_batch)
        assert mu.shape == (BATCH, LATENT_DIM), f"mu shape: {mu.shape}"
        assert logvar.shape == (BATCH, LATENT_DIM), f"logvar shape: {logvar.shape}"

    def test_mu_logvar_are_different(self, sample_batch: torch.Tensor):
        encoder = Encoder()
        mu, logvar = encoder(sample_batch)
        # They should not be identical (different linear heads)
        assert not torch.allclose(mu, logvar), "mu and logvar should differ"


# ──────────────────────────────────────────────
# Decoder tests
# ──────────────────────────────────────────────
class TestDecoder:
    def test_output_shape(self):
        decoder = Decoder()
        z = torch.randn(BATCH, LATENT_DIM)
        out = decoder(z)
        assert out.shape == (BATCH, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), (
            f"Decoder output shape: {out.shape}"
        )

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        decoder = Decoder()
        z = torch.randn(BATCH, LATENT_DIM)
        out = decoder(z)
        assert out.min() >= 0.0, f"Min pixel value {out.min().item()}"
        assert out.max() <= 1.0, f"Max pixel value {out.max().item()}"


# ──────────────────────────────────────────────
# Full VAE tests
# ──────────────────────────────────────────────
class TestVAE:
    def test_forward_shapes(self, vae_model: VAE, sample_batch: torch.Tensor):
        with torch.no_grad():
            recon, mu, logvar = vae_model(sample_batch)
        assert recon.shape == sample_batch.shape, f"recon shape: {recon.shape}"
        assert mu.shape == (BATCH, LATENT_DIM)
        assert logvar.shape == (BATCH, LATENT_DIM)

    def test_reconstruction_range(self, vae_model: VAE, sample_batch: torch.Tensor):
        with torch.no_grad():
            recon, _, _ = vae_model(sample_batch)
        assert recon.min() >= 0.0 and recon.max() <= 1.0

    def test_encode_convenience(self, vae_model: VAE, sample_batch: torch.Tensor):
        with torch.no_grad():
            mu, logvar = vae_model.encode(sample_batch)
        assert mu.shape == (BATCH, LATENT_DIM)
        assert logvar.shape == (BATCH, LATENT_DIM)

    def test_decode_convenience(self, vae_model: VAE):
        z = torch.randn(BATCH, LATENT_DIM)
        with torch.no_grad():
            recon = vae_model.decode(z)
        assert recon.shape == (BATCH, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    def test_reparameterize_stochastic(self):
        """Two calls with the same mu/logvar should (almost surely) differ."""
        mu = torch.zeros(BATCH, LATENT_DIM)
        logvar = torch.zeros(BATCH, LATENT_DIM)
        z1 = VAE.reparameterize(mu, logvar)
        z2 = VAE.reparameterize(mu, logvar)
        assert not torch.allclose(z1, z2), "Reparameterize should be stochastic"

    def test_reparameterize_deterministic_at_zero_var(self):
        """With logvar → -inf (var → 0), z should ≈ mu."""
        mu = torch.ones(BATCH, LATENT_DIM) * 3.0
        logvar = torch.full((BATCH, LATENT_DIM), -100.0)  # exp(-50) ≈ 0
        z = VAE.reparameterize(mu, logvar)
        assert torch.allclose(z, mu, atol=1e-6), "z should equal mu when var → 0"

    def test_custom_latent_dim(self, sample_batch: torch.Tensor):
        model = VAE(latent_dim=32)
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(sample_batch)
        assert mu.shape == (BATCH, 32)
        assert recon.shape == sample_batch.shape

    def test_gradients_flow(self, sample_batch: torch.Tensor):
        """Verify that gradients propagate through the full model."""
        model = VAE()
        model.train()
        recon, mu, logvar = model(sample_batch)
        loss, _, _ = vae_loss(recon, sample_batch, mu, logvar)
        loss.backward()
        # Check at least one encoder and one decoder param has a gradient
        enc_grads = [p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.encoder.parameters()]
        dec_grads = [p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.decoder.parameters()]
        assert any(enc_grads), "No gradient reached the encoder"
        assert any(dec_grads), "No gradient reached the decoder"


# ──────────────────────────────────────────────
# Loss function tests
# ──────────────────────────────────────────────
class TestLosses:
    def test_reconstruction_loss_zero(self):
        """MSE of identical tensors should be 0."""
        x = torch.rand(BATCH, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        loss = reconstruction_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_reconstruction_loss_positive(self):
        x = torch.rand(BATCH, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        y = torch.rand(BATCH, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        loss = reconstruction_loss(x, y)
        assert loss.item() > 0.0

    def test_kl_zero_at_prior(self):
        """KL should be 0 when q = N(0, I) (mu=0, logvar=0)."""
        mu = torch.zeros(BATCH, LATENT_DIM)
        logvar = torch.zeros(BATCH, LATENT_DIM)
        kl = kl_divergence(mu, logvar)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_positive_when_shifted(self):
        """KL should be > 0 when mu ≠ 0."""
        mu = torch.ones(BATCH, LATENT_DIM) * 2.0
        logvar = torch.zeros(BATCH, LATENT_DIM)
        kl = kl_divergence(mu, logvar)
        assert kl.item() > 0.0

    def test_vae_loss_returns_three_scalars(self, sample_batch: torch.Tensor):
        model = VAE()
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(sample_batch)
        total, recon_l, kl_l = vae_loss(recon, sample_batch, mu, logvar)
        assert total.dim() == 0, "total_loss should be scalar"
        assert recon_l.dim() == 0, "recon_loss should be scalar"
        assert kl_l.dim() == 0, "kl_loss should be scalar"

    def test_kl_weight_scales_total(self, sample_batch: torch.Tensor):
        model = VAE()
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(sample_batch)
        total_1, r1, k1 = vae_loss(recon, sample_batch, mu, logvar, kl_weight=1.0)
        total_5, r5, k5 = vae_loss(recon, sample_batch, mu, logvar, kl_weight=5.0)
        # Reconstruction component should be the same
        assert r1.item() == pytest.approx(r5.item(), abs=1e-7)
        # Total with higher β should be larger (assuming KL > 0)
        if k1.item() > 1e-8:
            assert total_5.item() > total_1.item()


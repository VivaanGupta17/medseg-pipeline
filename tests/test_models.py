"""Tests for U-Net and Attention U-Net model architectures."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.unet import UNet, ConvBlock, EncoderBlock, DecoderBlock
from models.attention_unet import AttentionUNet, AttentionGate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unet_2d():
    return UNet(in_channels=4, num_classes=4, spatial_dims=2,
                features=(16, 32, 64, 128))

@pytest.fixture
def unet_3d():
    return UNet(in_channels=1, num_classes=2, spatial_dims=3,
                features=(8, 16, 32))

@pytest.fixture
def attn_unet_2d():
    return AttentionUNet(in_channels=4, num_classes=4, spatial_dims=2,
                         features=(16, 32, 64))


# ---------------------------------------------------------------------------
# ConvBlock tests
# ---------------------------------------------------------------------------

class TestConvBlock:
    def test_output_shape_2d(self):
        block = ConvBlock(in_channels=4, out_channels=64, spatial_dims=2)
        x = torch.randn(2, 4, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_output_shape_3d(self):
        block = ConvBlock(in_channels=1, out_channels=32, spatial_dims=3)
        x = torch.randn(1, 1, 16, 16, 16)
        out = block(x)
        assert out.shape == (1, 32, 16, 16, 16)

    def test_residual_connection(self):
        block = ConvBlock(in_channels=4, out_channels=64, spatial_dims=2, residual=True)
        x = torch.randn(2, 4, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_dropout(self):
        block = ConvBlock(in_channels=4, out_channels=64, spatial_dims=2, dropout_p=0.5)
        # Check dropout module is present
        has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
                          for m in block.modules())
        assert has_dropout


# ---------------------------------------------------------------------------
# UNet tests
# ---------------------------------------------------------------------------

class TestUNet:
    def test_forward_2d_brats(self, unet_2d):
        """Full forward pass for 2-D BraTS-style input."""
        x = torch.randn(2, 4, 128, 128)
        out = unet_2d(x)
        assert out.shape == (2, 4, 128, 128), f"Expected (2,4,128,128) got {out.shape}"

    def test_forward_preserves_spatial_size(self):
        """Output spatial size must equal input regardless of odd dimensions."""
        model = UNet(in_channels=1, num_classes=2, spatial_dims=2, features=(16, 32))
        for h, w in [(64, 64), (128, 128), (100, 100), (63, 63)]:
            x = torch.randn(1, 1, h, w)
            out = model(x)
            assert out.shape == (1, 2, h, w), \
                f"Shape mismatch for ({h},{w}): got {out.shape}"

    def test_forward_3d(self, unet_3d):
        x = torch.randn(1, 1, 32, 32, 32)
        out = unet_3d(x)
        assert out.shape == (1, 2, 32, 32, 32)

    def test_parameter_count_reasonable(self, unet_2d):
        n_params = unet_2d.count_parameters()
        assert n_params > 0
        assert n_params < 50_000_000  # <50M for test config

    def test_output_is_raw_logits(self, unet_2d):
        """Output should NOT have softmax applied (raw logits expected)."""
        x = torch.randn(2, 4, 64, 64)
        out = unet_2d(x)
        # Raw logits can have negative values
        assert out.min() < 0 or out.max() > 1, "Output looks like probabilities, expected logits"

    def test_gradient_flow(self, unet_2d):
        x = torch.randn(2, 4, 64, 64)
        out = unet_2d(x)
        loss = out.sum()
        loss.backward()
        for name, param in unet_2d.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_instance_norm(self):
        model = UNet(in_channels=1, num_classes=2, spatial_dims=2,
                     features=(16, 32), norm_type="instance")
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 2, 64, 64)

    def test_invalid_spatial_dims(self):
        with pytest.raises(ValueError, match="spatial_dims"):
            UNet(spatial_dims=4)

    def test_mc_dropout_uncertainty(self):
        model = UNet(in_channels=1, num_classes=2, spatial_dims=2,
                     features=(16,), dropout_p=0.5)
        x = torch.randn(1, 1, 32, 32)
        mean, var = model.predict_with_uncertainty(x, n_passes=5)
        assert mean.shape == (1, 2, 32, 32)
        assert var.shape == (1, 2, 32, 32)
        assert (var >= 0).all()


# ---------------------------------------------------------------------------
# AttentionUNet tests
# ---------------------------------------------------------------------------

class TestAttentionUNet:
    def test_forward_shape(self, attn_unet_2d):
        x = torch.randn(2, 4, 128, 128)
        out = attn_unet_2d(x)
        assert out.shape == (2, 4, 128, 128)

    def test_attention_maps_populated(self, attn_unet_2d):
        x = torch.randn(1, 4, 64, 64)
        out = attn_unet_2d(x)
        maps = attn_unet_2d.get_attention_maps()
        assert len(maps) == 3, f"Expected 3 attention maps (depth=3), got {len(maps)}"
        for m in maps:
            assert m.shape[0] == 1       # batch dim
            assert m.shape[1] == 1       # single-channel attention

    def test_attention_values_in_range(self, attn_unet_2d):
        x = torch.randn(1, 4, 64, 64)
        attn_unet_2d(x)
        for m in attn_unet_2d.attention_maps:
            # Attention weights should be in [0, 1] (sigmoid output)
            assert m.min() >= 0.0 - 1e-6
            assert m.max() <= 1.0 + 1e-6

    def test_no_forward_raises(self):
        model = AttentionUNet(in_channels=1, num_classes=2, features=(8,))
        with pytest.raises(RuntimeError):
            model.get_attention_maps()

    def test_gradient_flow(self, attn_unet_2d):
        x = torch.randn(2, 4, 64, 64)
        out = attn_unet_2d(x)
        out.sum().backward()
        for name, param in attn_unet_2d.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# AttentionGate tests
# ---------------------------------------------------------------------------

class TestAttentionGate:
    def test_output_shape_2d(self):
        gate = AttentionGate(x_channels=64, g_channels=128, spatial_dims=2)
        x = torch.randn(2, 64, 32, 32)
        g = torch.randn(2, 128, 32, 32)
        out, alpha = gate(x, g)
        assert out.shape == x.shape
        assert alpha.shape == (2, 1, 32, 32)

    def test_alpha_is_sigmoidal(self):
        gate = AttentionGate(x_channels=32, g_channels=64, spatial_dims=2)
        x = torch.randn(1, 32, 16, 16)
        g = torch.randn(1, 64, 16, 16)
        _, alpha = gate(x, g)
        assert alpha.min() >= 0.0 - 1e-6
        assert alpha.max() <= 1.0 + 1e-6

    def test_gating_upsampling(self):
        """Gate signal at coarser resolution should be upsampled."""
        gate = AttentionGate(x_channels=64, g_channels=128, spatial_dims=2)
        x = torch.randn(1, 64, 32, 32)
        g = torch.randn(1, 128, 16, 16)  # coarser
        out, alpha = gate(x, g)
        assert out.shape == x.shape
        assert alpha.shape == (1, 1, 32, 32)

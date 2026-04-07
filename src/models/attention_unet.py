"""Attention U-Net for medical image segmentation.

Reference:
    Oktay, O., et al. (2018).
    Attention U-Net: Learning Where to Look for the Pancreas.
    MIDL 2018. https://arxiv.org/abs/1804.03999

The key addition over standard U-Net is **Additive Attention Gates** inserted
at each skip connection. The gate suppresses activations in irrelevant regions
and highlights the target structure, producing soft spatial attention maps that
can be visualised for model transparency.

This implementation extends ``UNet`` by swapping the plain skip-concatenation
in each decoder block with an attention-gated skip connection.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import ConvBlock, UNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention Gate
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    Computes a soft attention map from a gating signal ``g`` (coarser, from the
    decoder path) and a skip connection ``x`` (finer, from the encoder path).
    The map is learned end-to-end and used to re-weight the skip features.

    Architecture::

        x (skip)  ──  Wx  ──┐
                             + ─ ReLU ─ Wx ─ Sigmoid ─ α
        g (gate)  ──  Wg  ──┘
                                                         │
        x ─────────────────────────────────────── ×  ───→  x_hat

    Args:
        x_channels: Number of channels in the skip connection.
        g_channels: Number of channels in the gating signal.
        inter_channels: Intermediate channel dimension. Defaults to
            ``x_channels // 2``.
        spatial_dims: 2 or 3.
    """

    def __init__(
        self,
        x_channels: int,
        g_channels: int,
        inter_channels: Optional[int] = None,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()

        if inter_channels is None:
            inter_channels = max(x_channels // 2, 1)

        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d

        # Linear transformations
        self.Wx = Conv(x_channels, inter_channels, kernel_size=1, bias=True)
        self.Wg = Conv(g_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = Conv(inter_channels, 1, kernel_size=1, bias=True)

        self.bn = (nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d)(inter_channels)

        self.spatial_dims = spatial_dims

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-gated skip features.

        Args:
            x: Skip connection tensor, shape ``(B, x_channels, *spatial)``.
            g: Gating signal from decoder, shape ``(B, g_channels, *spatial_g)``.
               May be spatially smaller than ``x``; it is upsampled to match.

        Returns:
            Tuple of:
                - Attended skip features, same shape as ``x``.
                - Attention map ``alpha``, shape ``(B, 1, *spatial)``.
        """
        # Upsample gating signal to match skip spatial resolution
        if g.shape[2:] != x.shape[2:]:
            mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
            g = F.interpolate(g, size=x.shape[2:], mode=mode, align_corners=False)

        theta_x = self.Wx(x)
        phi_g = self.Wg(g)

        # Additive attention
        f = F.relu(self.bn(theta_x + phi_g), inplace=True)
        alpha = torch.sigmoid(self.psi(f))  # (B, 1, *spatial)

        return x * alpha, alpha


# ---------------------------------------------------------------------------
# Attention Decoder Block
# ---------------------------------------------------------------------------

class AttentionDecoderBlock(nn.Module):
    """Decoder block with an attention gate on the skip connection.

    Args:
        in_channels: Channels of the upsampled tensor (from below).
        skip_channels: Channels of the skip connection.
        out_channels: Output channels after the conv block.
        spatial_dims: 2 or 3.
        **conv_kwargs: Forwarded to ``ConvBlock``.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        spatial_dims: int = 2,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims

        ConvTranspose = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        self.upsample = ConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.attention_gate = AttentionGate(
            x_channels=skip_channels,
            g_channels=in_channels // 2,
            spatial_dims=spatial_dims,
        )

        self.conv_block = ConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            spatial_dims,
            **conv_kwargs,
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor from the level below.
            skip: Skip connection tensor from the encoder.

        Returns:
            Tuple of (output tensor, attention map for visualisation).
        """
        x = self.upsample(x)

        # Handle size mismatches
        if x.shape[2:] != skip.shape[2:]:
            diff = [skip.size(i) - x.size(i) for i in range(2, x.ndim)]
            pad = []
            for d in reversed(diff):
                pad += [0, d]
            x = F.pad(x, pad)

        attended_skip, alpha = self.attention_gate(skip, x)
        x = torch.cat([attended_skip, x], dim=1)
        return self.conv_block(x), alpha


# ---------------------------------------------------------------------------
# Attention U-Net
# ---------------------------------------------------------------------------

class AttentionUNet(nn.Module):
    """Attention U-Net for medical image segmentation.

    Identical to standard U-Net but with attention gates at every skip
    connection. Attention maps are accessible via ``self.attention_maps``
    after each forward pass for visualisation and interpretability.

    Args:
        in_channels: Number of input modalities/channels.
        num_classes: Number of segmentation classes including background.
        spatial_dims: 2 or 3.
        features: Tuple of channel widths per encoder level.
        norm_type: ``'batch'`` or ``'instance'``.
        dropout_p: Dropout probability (enables MC Dropout when > 0).
        residual: Whether to use residual connections in ConvBlocks.

    Example::

        model = AttentionUNet(in_channels=4, num_classes=4, spatial_dims=2)
        x = torch.randn(2, 4, 240, 240)
        logits = model(x)           # (2, 4, 240, 240)

        # Inspect attention maps from last forward pass
        for level, alpha in enumerate(model.attention_maps):
            print(f"Level {level}: alpha shape = {alpha.shape}")
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        spatial_dims: int = 2,
        features: Tuple[int, ...] = (64, 128, 256, 512),
        norm_type: str = "batch",
        dropout_p: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims
        self.features = features

        # Attention maps populated during forward pass (for visualisation)
        self.attention_maps: list[torch.Tensor] = []

        norm = UNet._get_norm(norm_type, spatial_dims)
        conv_kwargs = dict(norm=norm, dropout_p=dropout_p, residual=residual)

        # ── Encoder (identical to UNet) ──────────────────────────────────────
        from .unet import EncoderBlock
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in features:
            self.encoders.append(EncoderBlock(prev_ch, ch, spatial_dims, **conv_kwargs))
            prev_ch = ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_ch = features[-1] * 2
        self.bottleneck = ConvBlock(prev_ch, bottleneck_ch, spatial_dims, **conv_kwargs)

        # ── Attention Decoder ────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        up_ch = bottleneck_ch
        for ch in reversed(features):
            self.decoders.append(
                AttentionDecoderBlock(up_ch, ch, ch, spatial_dims, **conv_kwargs)
            )
            up_ch = ch

        # ── Segmentation head ────────────────────────────────────────────────
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.seg_head = Conv(features[0], num_classes, kernel_size=1)

        self._init_weights()
        logger.info(
            "Initialized AttentionUNet: spatial_dims=%d  depth=%d  features=%s  params=%.2fM",
            spatial_dims, len(features), features, self.count_parameters() / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Populates ``self.attention_maps`` (list of tensors, one per decoder level)
        as a side effect. Useful for post-hoc visualisation.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` or ``(B, C, D, H, W)``.

        Returns:
            Raw logits of the same spatial size as the input.
        """
        self.attention_maps = []

        # Encoder
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Attention decoder
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x, alpha = decoder(x, skip)
            self.attention_maps.append(alpha.detach())

        return self.seg_head(x)

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Return attention maps from the most recent forward pass.

        Returns:
            List of attention tensors from shallowest to deepest decoder level.
            Each tensor has shape ``(B, 1, *spatial)``.
        """
        if not self.attention_maps:
            raise RuntimeError("No forward pass has been performed yet.")
        return self.attention_maps

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "AttentionUNet":
        """Load a pre-trained AttentionUNet from a checkpoint file.

        Args:
            checkpoint_path: Path to ``.pth`` checkpoint containing
                ``{'model_state_dict': ..., 'config': ...}``.
            **kwargs: Override config values.

        Returns:
            Loaded model in eval mode.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = {**checkpoint.get("config", {}), **kwargs}
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("Loaded AttentionUNet from %s", checkpoint_path)
        return model

    def enable_mc_dropout(self) -> None:
        """Activate only Dropout layers for MC Dropout uncertainty estimation."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_passes: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout uncertainty estimation.

        Args:
            x: Input tensor.
            n_passes: Number of stochastic forward passes.

        Returns:
            Tuple of (mean softmax predictions, predictive variance).
        """
        self.enable_mc_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(F.softmax(self.forward(x), dim=1))

        stacked = torch.stack(preds, dim=0)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0)
        return mean, variance

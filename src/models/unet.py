"""U-Net architecture for medical image segmentation.

Reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-net: Convolutional networks for biomedical image segmentation.
    MICCAI 2015. https://arxiv.org/abs/1505.04597

Supports:
    - 2D and 3D variants (controlled via `spatial_dims`)
    - Configurable depth and channel widths
    - Batch Normalization or Instance Normalization
    - Dropout for MC uncertainty estimation
    - Residual connections (optional)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two successive (Conv → Norm → ReLU) layers.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        spatial_dims: 2 for 2-D inputs, 3 for 3-D inputs.
        norm: Normalisation layer class, e.g. ``nn.BatchNorm2d``.
        dropout_p: Dropout probability applied between the two convolutions.
        residual: If True, add a learnable residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 2,
        norm: Optional[Type[nn.Module]] = None,
        dropout_p: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()

        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        Dropout = nn.Dropout2d if spatial_dims == 2 else nn.Dropout3d

        if norm is None:
            norm = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d

        self.residual = residual

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm(out_channels)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        if residual:
            self.skip_conv = (
                Conv(in_channels, out_channels, kernel_size=1, bias=False)
                if in_channels != out_channels
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.act(self.norm2(self.conv2(out)))

        if self.residual:
            out = out + self.skip_conv(identity)

        return out


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock followed by max-pool downsampling.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        spatial_dims: Spatial dimensionality.
        pool_size: Kernel size (and stride) for max pooling.
        **conv_kwargs: Forwarded to ``ConvBlock``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 2,
        pool_size: int = 2,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, spatial_dims, **conv_kwargs)
        MaxPool = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        self.pool = MaxPool(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (downsampled tensor, skip-connection tensor).
        """
        skip = self.conv_block(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Decoder block: bilinear upsample + skip concatenation + ConvBlock.

    Args:
        in_channels: Channels of the upsampled input (from below).
        skip_channels: Channels of the skip connection (from encoder).
        out_channels: Output channels after the conv block.
        spatial_dims: Spatial dimensionality.
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
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d

        # Learnable upsampling via transposed convolution
        ConvTranspose = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        self.upsample = ConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv_block = ConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            spatial_dims,
            **conv_kwargs,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatches from odd input dimensions
        if x.shape != skip.shape:
            diff = [skip.size(i) - x.size(i) for i in range(2, x.ndim)]
            # F.pad expects (last_dim_left, last_dim_right, ...) in reverse order
            pad = []
            for d in reversed(diff):
                pad += [0, d]
            x = F.pad(x, pad)

        x = torch.cat([skip, x], dim=1)
        return self.conv_block(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """Configurable U-Net for 2-D or 3-D medical image segmentation.

    Architecture follows the original Ronneberger et al. (2015) design with
    modern improvements: batch normalization, LeakyReLU, and optional residual
    skip connections.

    Args:
        in_channels: Number of input modalities/channels (e.g. 4 for BraTS).
        num_classes: Number of segmentation classes including background.
        spatial_dims: 2 or 3.
        features: Tuple of channel widths for each encoder depth level.
            Length determines network depth. Default ``(64, 128, 256, 512)``
            gives a 4-level encoder and 1 bottleneck.
        norm_type: ``'batch'`` or ``'instance'``.
        dropout_p: Dropout probability. Set > 0 to enable MC Dropout inference.
        residual: Whether to use residual connections in ConvBlocks.

    Example::

        model = UNet(in_channels=4, num_classes=4, spatial_dims=2)
        x = torch.randn(2, 4, 240, 240)
        logits = model(x)          # (2, 4, 240, 240)
        probs = logits.softmax(1)  # class probabilities
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
        if norm_type not in ("batch", "instance"):
            raise ValueError(f"norm_type must be 'batch' or 'instance', got {norm_type}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_dims = spatial_dims
        self.features = features

        norm = self._get_norm(norm_type, spatial_dims)
        conv_kwargs = dict(norm=norm, dropout_p=dropout_p, residual=residual)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in features:
            self.encoders.append(
                EncoderBlock(prev_ch, ch, spatial_dims, **conv_kwargs)
            )
            prev_ch = ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        bottleneck_ch = features[-1] * 2
        self.bottleneck = ConvBlock(prev_ch, bottleneck_ch, spatial_dims, **conv_kwargs)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        up_ch = bottleneck_ch
        for ch in reversed(features):
            self.decoders.append(
                DecoderBlock(up_ch, ch, ch, spatial_dims, **conv_kwargs)
            )
            up_ch = ch

        # ── Segmentation head ────────────────────────────────────────────────
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.seg_head = Conv(features[0], num_classes, kernel_size=1)

        self._init_weights()
        logger.info(
            "Initialized UNet: spatial_dims=%d  depth=%d  features=%s  params=%.2fM",
            spatial_dims, len(features), features, self.count_parameters() / 1e6,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` or ``(B, C, D, H, W)``.

        Returns:
            Raw logits of the same spatial size as input,
            shape ``(B, num_classes, H, W)`` or ``(B, num_classes, D, H, W)``.
        """
        # Encoder
        skips: List[torch.Tensor] = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.seg_head(x)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_norm(norm_type: str, spatial_dims: int) -> Type[nn.Module]:
        """Return appropriate normalisation layer class."""
        mapping = {
            ("batch", 2): nn.BatchNorm2d,
            ("batch", 3): nn.BatchNorm3d,
            ("instance", 2): nn.InstanceNorm2d,
            ("instance", 3): nn.InstanceNorm3d,
        }
        return mapping[(norm_type, spatial_dims)]

    def _init_weights(self) -> None:
        """Kaiming initialisation for conv layers; constant init for norms."""
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
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "UNet":
        """Load a pre-trained UNet from a checkpoint file.

        The checkpoint should contain ``{'model_state_dict': ..., 'config': ...}``.

        Args:
            checkpoint_path: Path to the ``.pth`` checkpoint.
            **kwargs: Override config values from the checkpoint.

        Returns:
            Loaded UNet in eval mode.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = {**checkpoint.get("config", {}), **kwargs}
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("Loaded UNet from %s", checkpoint_path)
        return model

    def enable_mc_dropout(self) -> None:
        """Set model to train mode only for Dropout layers (MC Dropout)."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_passes: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run MC Dropout inference to estimate epistemic uncertainty.

        Args:
            x: Input tensor.
            n_passes: Number of stochastic forward passes.

        Returns:
            Tuple of (mean prediction logits, variance map).
        """
        if not any(
            isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)) for m in self.modules()
        ):
            logger.warning(
                "No dropout layers found. Set dropout_p > 0 during model init for "
                "meaningful uncertainty estimates."
            )

        self.enable_mc_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(F.softmax(self.forward(x), dim=1))

        stacked = torch.stack(preds, dim=0)  # (n_passes, B, C, ...)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0)
        return mean, variance

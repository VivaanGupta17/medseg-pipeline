"""Loss functions for medical image segmentation.

Implements:
    - Dice loss (soft, with optional log form)
    - Focal loss (for class imbalance in detection-style settings)
    - Tversky loss (asymmetric weighting of FP/FN, suited for imbalanced data)
    - Combined Dice + Cross-Entropy loss (standard for brain tumour segmentation)

All losses operate on logits (unnormalised model outputs) of shape
``(B, C, *spatial)`` and integer targets of shape ``(B, *spatial)``.

References:
    Milletari et al. (2016). V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation.

    Salehi et al. (2017). Tversky Loss Function for Image Segmentation Using
    3D Fully Convolutional Deep Networks.

    Lin et al. (2017). Focal Loss for Dense Object Detection.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(
    targets: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert integer label map to one-hot encoding.

    Args:
        targets: ``(B, *spatial)`` integer tensor.
        num_classes: Number of classes.
        dtype: Output dtype.

    Returns:
        ``(B, C, *spatial)`` one-hot tensor.
    """
    spatial = targets.shape[1:]
    # (B, *spatial) → (B, 1, *spatial) → (B, C, *spatial)
    one_hot = torch.zeros(
        targets.shape[0], num_classes, *spatial,
        dtype=dtype, device=targets.device,
    )
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    return one_hot


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Computes 1 - Dice per class, then optionally averages.

    .. math::

        \\mathcal{L}_{Dice} = 1 - \\frac{2 \\sum p \\cdot t + \\epsilon}
                                        {\\sum p + \\sum t + \\epsilon}

    Args:
        num_classes: Number of output classes.
        ignore_index: Class index to ignore (e.g. background). Use ``-1``
            to include all classes.
        log_loss: If True, compute ``-log(Dice)`` for numerically stable
            gradients in early training.
        smooth: Laplace smoothing constant.
        weight: Per-class weights tensor of shape ``(num_classes,)``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        ignore_index: int = -1,
        log_loss: bool = False,
        smooth: float = 1.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.log_loss = log_loss
        self.smooth = smooth
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits: Raw model output ``(B, C, *spatial)``.
            targets: Integer labels ``(B, *spatial)``.

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = _one_hot(targets, self.num_classes, dtype=probs.dtype)

        # Flatten spatial dimensions: (B, C, N)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets_oh.view(targets_oh.size(0), targets_oh.size(1), -1)

        # Dice per class: sum over batch and spatial dims
        intersection = (probs_flat * targets_flat).sum(dim=(0, 2))
        denominator = probs_flat.sum(dim=(0, 2)) + targets_flat.sum(dim=(0, 2))

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # log-dice has better-behaved gradients early in training when dice ≈ 0
        if self.log_loss:
            loss_per_class = -torch.log(dice.clamp_min(1e-7))
        else:
            loss_per_class = 1.0 - dice

        if self.ignore_index >= 0:
            mask = torch.ones(self.num_classes, device=logits.device, dtype=torch.bool)
            mask[self.ignore_index] = False
            loss_per_class = loss_per_class[mask]

        if self.weight is not None:
            w = self.weight.to(logits.device)
            if self.ignore_index >= 0:
                idx = [i for i in range(self.num_classes) if i != self.ignore_index]
                w = w[idx]
            loss_per_class = loss_per_class * w / w.sum()

        return loss_per_class.mean()


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in segmentation.

    .. math::

        \\mathcal{L}_{focal} = -\\alpha_t (1 - p_t)^{\\gamma} \\log(p_t)

    where ``p_t`` is the predicted probability for the true class.

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples.
        alpha: Class balancing weights. Scalar or tensor of shape ``(C,)``.
        ignore_index: Class index to exclude from loss.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(B, C, *spatial)``
            targets: ``(B, *spatial)``

        Returns:
            Scalar (or per-element if reduction='none') loss.
        """
        # Flatten spatial dims: (B*N, C) and (B*N,)
        B, C = logits.shape[:2]
        logits_flat = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        log_probs = F.log_softmax(logits_flat, dim=-1)           # (N, C)
        probs     = log_probs.exp()

        # Gather log-probs and probs for the true class
        valid = targets_flat != self.ignore_index
        log_pt = log_probs[valid].gather(1, targets_flat[valid].unsqueeze(1)).squeeze(1)
        pt     = probs[valid].gather(1, targets_flat[valid].unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Tversky Loss
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """Tversky loss for handling severe class imbalance.

    Generalises Dice loss by independently weighting false positives and
    false negatives:

    .. math::

        T = \\frac{\\sum p \\cdot t + \\epsilon}
                  {\\sum p \\cdot t + \\alpha \\sum p(1-t)
                   + \\beta \\sum (1-p)t + \\epsilon}

    With ``alpha=beta=0.5`` this reduces to Dice loss.
    Setting ``alpha=0.3, beta=0.7`` penalises false negatives more,
    useful for finding small lesions.

    Args:
        alpha: Weight for false positives.
        beta: Weight for false negatives.
        num_classes: Number of output classes.
        ignore_index: Class index to ignore.
        smooth: Smoothing constant.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        num_classes: int = 4,
        ignore_index: int = -1,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(f"alpha + beta must equal 1.0, got {alpha + beta}")
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss.

        Args:
            logits: ``(B, C, *spatial)``
            targets: ``(B, *spatial)``

        Returns:
            Scalar loss.
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = _one_hot(targets, self.num_classes, dtype=probs.dtype)

        probs_flat   = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets_oh.view(targets_oh.size(0), targets_oh.size(1), -1)

        tp = (probs_flat * targets_flat).sum(dim=(0, 2))
        fp = (probs_flat * (1 - targets_flat)).sum(dim=(0, 2))
        fn = ((1 - probs_flat) * targets_flat).sum(dim=(0, 2))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss_per_class = 1.0 - tversky

        if self.ignore_index >= 0:
            mask = torch.ones(self.num_classes, device=logits.device, dtype=torch.bool)
            mask[self.ignore_index] = False
            loss_per_class = loss_per_class[mask]

        return loss_per_class.mean()


# ---------------------------------------------------------------------------
# Combined Dice + Cross-Entropy Loss
# ---------------------------------------------------------------------------

class CombinedDiceCELoss(nn.Module):
    """Weighted combination of Dice loss and Cross-Entropy loss.

    This is the standard loss for BraTS segmentation. Cross-entropy provides
    stable per-voxel gradients early in training, while Dice loss directly
    optimises the evaluation metric.

    .. math::

        \\mathcal{L} = \\lambda_{Dice} \\mathcal{L}_{Dice}
                      + \\lambda_{CE} \\mathcal{L}_{CE}

    Args:
        num_classes: Number of output classes.
        dice_weight: Contribution weight of Dice loss.
        ce_weight: Contribution weight of Cross-Entropy loss.
        ignore_index: Class to ignore in CE (and optionally Dice).
        class_weights: Per-class weights for Cross-Entropy (handles imbalance).
        dice_log: Use log-Dice for smoother gradients.
        label_smoothing: Label smoothing factor for CE (regularisation).
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
        dice_log: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index if ignore_index >= 0 else -1,
            log_loss=dice_log,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Args:
            logits: ``(B, C, *spatial)``
            targets: ``(B, *spatial)``

        Returns:
            Tuple of (total_loss, dice_loss_value, ce_loss_value).
        """
        dice = self.dice_loss(logits, targets)
        ce   = self.ce_loss(logits, targets)
        total = self.dice_weight * dice + self.ce_weight * ce
        return total, dice, ce

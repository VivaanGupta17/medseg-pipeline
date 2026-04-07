"""Tests for segmentation loss functions."""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.losses import DiceLoss, FocalLoss, TverskyLoss, CombinedDiceCELoss


def make_perfect_prediction(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create one-hot logits that would give perfect predictions."""
    B = targets.shape[0]
    spatial = targets.shape[1:]
    logits = torch.full((B, num_classes, *spatial), -10.0)
    for b in range(B):
        for c in range(num_classes):
            logits[b, c][targets[b] == c] = 10.0
    return logits


class TestDiceLoss:
    def test_perfect_prediction_gives_zero_loss(self):
        targets = torch.randint(0, 4, (2, 32, 32))
        logits = make_perfect_prediction(targets, 4)
        loss = DiceLoss(num_classes=4)(logits, targets)
        assert loss.item() == pytest.approx(0.0, abs=0.01)

    def test_loss_range(self):
        logits  = torch.randn(2, 4, 16, 16)
        targets = torch.randint(0, 4, (2, 16, 16))
        loss = DiceLoss(num_classes=4)(logits, targets)
        assert 0.0 <= loss.item() <= 1.5

    def test_backward(self):
        logits  = torch.randn(2, 4, 16, 16, requires_grad=True)
        targets = torch.randint(0, 4, (2, 16, 16))
        loss = DiceLoss(num_classes=4)(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_ignore_background(self):
        logits  = torch.randn(2, 4, 16, 16)
        targets = torch.randint(0, 4, (2, 16, 16))
        loss_with_bg    = DiceLoss(num_classes=4, ignore_index=-1)(logits, targets)
        loss_without_bg = DiceLoss(num_classes=4, ignore_index=0)(logits, targets)
        # They should differ (background class affects the loss)
        # Both should be valid scalars
        assert not torch.isnan(loss_with_bg)
        assert not torch.isnan(loss_without_bg)

    def test_log_loss(self):
        logits  = torch.randn(2, 4, 16, 16)
        targets = torch.randint(0, 4, (2, 16, 16))
        loss = DiceLoss(num_classes=4, log_loss=True)(logits, targets)
        assert loss.item() >= 0.0


class TestFocalLoss:
    def test_loss_scalar(self):
        logits  = torch.randn(2, 4, 16, 16)
        targets = torch.randint(0, 4, (2, 16, 16))
        loss = FocalLoss(gamma=2.0)(logits, targets)
        assert loss.ndim == 0  # scalar

    def test_backward(self):
        logits  = torch.randn(2, 4, 16, 16, requires_grad=True)
        targets = torch.randint(0, 4, (2, 16, 16))
        loss = FocalLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_gamma_zero_equals_ce(self):
        """Focal loss with gamma=0 and alpha=None should equal CE loss."""
        logits  = torch.randn(2, 4, 32, 32)
        targets = torch.randint(0, 4, (2, 32, 32))
        focal = FocalLoss(gamma=0.0, alpha=None)(logits, targets)
        ce    = torch.nn.CrossEntropyLoss()(logits, targets)
        assert focal.item() == pytest.approx(ce.item(), rel=1e-3)


class TestTverskyLoss:
    def test_valid_alpha_beta(self):
        with pytest.raises(ValueError):
            TverskyLoss(alpha=0.5, beta=0.3)  # doesn't sum to 1

    def test_dice_equivalent(self):
        """alpha=beta=0.5 should give same result as Dice loss."""
        logits  = torch.randn(2, 4, 16, 16)
        targets = torch.randint(0, 4, (2, 16, 16))
        tversky = TverskyLoss(alpha=0.5, beta=0.5, num_classes=4)(logits, targets)
        dice    = DiceLoss(num_classes=4)(logits, targets)
        assert abs(tversky.item() - dice.item()) < 0.05

    def test_high_beta_penalises_fn(self):
        """With beta=0.9, FN are heavily penalised → higher loss when FN is high."""
        targets = torch.zeros(1, 16, 16, dtype=torch.long)
        targets[0, 4:8, 4:8] = 1
        
        # All-background prediction → lots of FN
        logits_all_bg = torch.zeros(1, 4, 16, 16)
        logits_all_bg[:, 0] = 10.0  # predict background everywhere
        
        loss_high_beta = TverskyLoss(alpha=0.1, beta=0.9, num_classes=4)(
            logits_all_bg, targets
        )
        loss_low_beta = TverskyLoss(alpha=0.9, beta=0.1, num_classes=4)(
            logits_all_bg, targets
        )
        assert loss_high_beta.item() > loss_low_beta.item()


class TestCombinedDiceCELoss:
    def test_returns_three_values(self):
        logits  = torch.randn(2, 4, 16, 16)
        targets = torch.randint(0, 4, (2, 16, 16))
        total, dice, ce = CombinedDiceCELoss(num_classes=4)(logits, targets)
        assert total.ndim == 0
        assert dice.ndim == 0
        assert ce.ndim == 0

    def test_total_is_weighted_sum(self):
        logits  = torch.randn(2, 4, 32, 32)
        targets = torch.randint(0, 4, (2, 32, 32))
        w_dice, w_ce = 0.3, 0.7
        total, dice, ce = CombinedDiceCELoss(
            num_classes=4, dice_weight=w_dice, ce_weight=w_ce
        )(logits, targets)
        expected = w_dice * dice + w_ce * ce
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_backward(self):
        logits  = torch.randn(2, 4, 16, 16, requires_grad=True)
        targets = torch.randint(0, 4, (2, 16, 16))
        total, _, _ = CombinedDiceCELoss(num_classes=4)(logits, targets)
        total.backward()
        assert logits.grad is not None

    def test_all_losses_positive(self):
        logits  = torch.randn(4, 4, 64, 64)
        targets = torch.randint(0, 4, (4, 64, 64))
        total, dice, ce = CombinedDiceCELoss(num_classes=4)(logits, targets)
        assert total.item() >= 0
        assert dice.item() >= 0
        assert ce.item() >= 0

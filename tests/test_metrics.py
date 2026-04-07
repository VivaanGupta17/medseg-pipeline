"""Tests for medical segmentation evaluation metrics."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import (
    SegmentationMetrics,
    dice_coefficient,
    iou_score,
    sensitivity,
    specificity,
    precision_score,
    volume_similarity,
    hausdorff_distance_95,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Dice coefficient
# ---------------------------------------------------------------------------

class TestDiceCoefficient:
    def test_perfect_overlap(self):
        mask = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
        assert dice_coefficient(mask, mask) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = np.array([1, 0, 0, 0])
        tgt  = np.array([0, 0, 0, 1])
        d = dice_coefficient(pred.astype(bool), tgt.astype(bool))
        assert d == pytest.approx(0.0, abs=1e-4)

    def test_empty_prediction(self):
        pred = np.zeros((10, 10), dtype=bool)
        tgt  = np.ones((10, 10), dtype=bool)
        d = dice_coefficient(pred, tgt)
        assert 0.0 <= d <= 1.0

    def test_empty_both(self):
        pred = np.zeros((5, 5), dtype=bool)
        tgt  = np.zeros((5, 5), dtype=bool)
        d = dice_coefficient(pred, tgt)
        # Both empty: smooth → 1.0 (convention)
        assert d == pytest.approx(1.0, abs=1e-4)

    def test_partial_overlap(self):
        pred = np.array([1, 1, 1, 0, 0])
        tgt  = np.array([1, 1, 0, 0, 1])
        # Intersection = 2, sizes = 3, 3 → Dice = 2*2/(3+3) = 0.667
        d = dice_coefficient(pred.astype(bool), tgt.astype(bool))
        assert d == pytest.approx(2/3, abs=1e-3)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

class TestIoU:
    def test_perfect(self):
        mask = np.ones((5, 5), dtype=bool)
        assert iou_score(mask, mask) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = np.array([1, 0])
        tgt  = np.array([0, 1])
        assert iou_score(pred.astype(bool), tgt.astype(bool)) == pytest.approx(0.0, abs=1e-4)

    def test_dice_ge_iou(self):
        """Dice is always ≥ IoU for non-trivial cases."""
        rng = np.random.default_rng(0)
        pred = rng.integers(0, 2, (20, 20)).astype(bool)
        tgt  = rng.integers(0, 2, (20, 20)).astype(bool)
        d = dice_coefficient(pred, tgt)
        i = iou_score(pred, tgt)
        assert d >= i - 1e-6


# ---------------------------------------------------------------------------
# Sensitivity & Specificity
# ---------------------------------------------------------------------------

class TestSensitivitySpecificity:
    def test_perfect_sensitivity(self):
        pred = np.ones(10, dtype=bool)
        tgt  = np.ones(10, dtype=bool)
        assert sensitivity(pred, tgt) == pytest.approx(1.0, abs=1e-4)

    def test_zero_sensitivity(self):
        pred = np.zeros(10, dtype=bool)
        tgt  = np.ones(10, dtype=bool)
        s = sensitivity(pred, tgt)
        assert s == pytest.approx(0.0, abs=1e-3)

    def test_perfect_specificity(self):
        pred = np.zeros(10, dtype=bool)
        tgt  = np.zeros(10, dtype=bool)
        assert specificity(pred, tgt) == pytest.approx(1.0, abs=1e-4)

    def test_sensitivity_specificity_complement_on_balanced(self):
        # 50% TP, 50% TN → sensitivity = specificity = 1.0
        pred = np.array([1, 1, 0, 0], dtype=bool)
        tgt  = np.array([1, 1, 0, 0], dtype=bool)
        assert sensitivity(pred, tgt) == pytest.approx(1.0, abs=1e-4)
        assert specificity(pred, tgt) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Hausdorff Distance 95
# ---------------------------------------------------------------------------

class TestHausdorffDistance:
    def test_identical_masks(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[8:12, 8:12, 8:12] = True
        hd = hausdorff_distance_95(mask, mask)
        assert hd == pytest.approx(0.0, abs=0.5)

    def test_shifted_mask(self):
        pred = np.zeros((30, 30), dtype=bool)
        tgt  = np.zeros((30, 30), dtype=bool)
        pred[10:15, 10:15] = True
        tgt[12:17, 10:15] = True  # 2-voxel shift
        hd = hausdorff_distance_95(pred, tgt)
        assert hd >= 0.0
        assert hd < 10.0  # Should be small for 2-voxel shift

    def test_empty_mask_returns_nan(self):
        pred = np.zeros((10, 10), dtype=bool)
        tgt  = np.ones((10, 10), dtype=bool)
        hd = hausdorff_distance_95(pred, tgt)
        assert np.isnan(hd)

    def test_with_voxel_spacing(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[8:12, 8:12, 8:12] = True
        shifted = np.zeros_like(mask)
        shifted[10:14, 8:12, 8:12] = True

        hd_vox = hausdorff_distance_95(mask, shifted, voxel_spacing=None)
        hd_mm  = hausdorff_distance_95(mask, shifted, voxel_spacing=(2.0, 1.0, 1.0))
        # With 2mm z-spacing, distance in z is doubled
        assert hd_mm >= hd_vox


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        cm = confusion_matrix(labels, labels, num_classes=4)
        assert cm.shape == (4, 4)
        assert (cm == np.diag(np.bincount(labels, minlength=4))).all()

    def test_all_background(self):
        pred = np.zeros(100, dtype=np.int64)
        tgt  = np.zeros(100, dtype=np.int64)
        cm = confusion_matrix(pred, tgt, num_classes=4)
        assert cm[0, 0] == 100
        assert cm[1:, :].sum() == 0

    def test_shape(self):
        pred = np.random.randint(0, 4, (10, 10))
        tgt  = np.random.randint(0, 4, (10, 10))
        cm = confusion_matrix(pred, tgt, num_classes=4)
        assert cm.shape == (4, 4)
        assert cm.sum() == 100


# ---------------------------------------------------------------------------
# SegmentationMetrics (integration)
# ---------------------------------------------------------------------------

class TestSegmentationMetrics:
    @pytest.fixture
    def metrics(self):
        return SegmentationMetrics(num_classes=4, compute_hd95=False)

    def test_compute_all_returns_expected_keys(self, metrics):
        pred = np.zeros((20, 20, 20), dtype=np.int64)
        tgt  = np.zeros((20, 20, 20), dtype=np.int64)
        pred[5:10, 5:10, 5:10] = 1
        tgt[5:10, 5:10, 5:10] = 1

        result = metrics.compute_all(pred, tgt)
        for key in ("mean_dice", "mean_iou", "sensitivity", "specificity",
                    "precision", "volume_similarity",
                    "wt_dice", "tc_dice", "et_dice"):
            assert key in result, f"Missing key: {key}"

    def test_perfect_segmentation(self, metrics):
        mask = np.zeros((20, 20, 20), dtype=np.int64)
        mask[5:10, 5:10, 5:10] = 1
        mask[10:15, 5:10, 5:10] = 2
        mask[5:10, 10:15, 5:10] = 3

        result = metrics.compute_all(mask, mask)
        assert result["mean_dice"] == pytest.approx(1.0, abs=1e-3)
        assert result["sensitivity"] == pytest.approx(1.0, abs=1e-3)
        assert result["specificity"] == pytest.approx(1.0, abs=1e-3)

    def test_all_background_prediction(self, metrics):
        pred = np.zeros((10, 10, 10), dtype=np.int64)
        tgt  = np.zeros((10, 10, 10), dtype=np.int64)
        tgt[3:7, 3:7, 3:7] = 1

        result = metrics.compute_all(pred, tgt)
        # Class 1 Dice should be 0 (missed all foreground)
        assert result["class_dice"][1] == pytest.approx(0.0, abs=1e-3)

    def test_batch_metrics_averaged(self, metrics):
        preds   = [np.random.randint(0, 4, (20, 20)) for _ in range(5)]
        targets = [np.random.randint(0, 4, (20, 20)) for _ in range(5)]
        
        agg = metrics.compute_batch(np.stack(preds), np.stack(targets))
        assert "mean_dice" in agg
        assert 0.0 <= agg["mean_dice"] <= 1.0

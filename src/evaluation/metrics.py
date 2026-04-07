"""Medical image segmentation evaluation metrics.

Implements the full BraTS challenge evaluation suite:
    - Dice coefficient (per-class and mean)
    - IoU / Jaccard index
    - Hausdorff distance (95th percentile) in voxels and mm
    - Sensitivity (recall) and specificity
    - Precision
    - Volume similarity
    - Average surface distance

All functions accept numpy arrays and return scalar or dict outputs.
The ``SegmentationMetrics`` class provides a high-level API for batch evaluation.

References:
    BraTS 2021 challenge evaluation code:
    https://github.com/rachitsaluja/BraTS-2023-Metrics

    Taha & Hanbury (2015). Metrics for evaluating 3D medical image segmentation.
    BMC Medical Imaging.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from scipy.spatial.distance import directed_hausdorff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def dice_coefficient(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-7,
) -> float:
    """Compute binary Dice coefficient (F1 score).

    .. math::
        Dice = \\frac{2 |P \\cap T|}{|P| + |T|}

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        smooth: Laplace smoothing to avoid division by zero.

    Returns:
        Dice score in [0, 1].
    """
    pred = prediction.astype(bool).ravel()
    tgt  = target.astype(bool).ravel()
    intersection = np.logical_and(pred, tgt).sum()
    return float(2.0 * intersection + smooth) / float(pred.sum() + tgt.sum() + smooth)


def iou_score(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-7,
) -> float:
    """Compute Intersection over Union (Jaccard index).

    .. math::
        IoU = \\frac{|P \\cap T|}{|P \\cup T|}

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        smooth: Smoothing constant.

    Returns:
        IoU score in [0, 1].
    """
    pred = prediction.astype(bool).ravel()
    tgt  = target.astype(bool).ravel()
    intersection = np.logical_and(pred, tgt).sum()
    union = np.logical_or(pred, tgt).sum()
    return float(intersection + smooth) / float(union + smooth)


def sensitivity(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-7,
) -> float:
    """Compute sensitivity (recall / true positive rate).

    Fraction of ground truth positives correctly identified.

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        smooth: Smoothing constant.

    Returns:
        Sensitivity in [0, 1].
    """
    pred = prediction.astype(bool).ravel()
    tgt  = target.astype(bool).ravel()
    tp = np.logical_and(pred, tgt).sum()
    fn = np.logical_and(~pred, tgt).sum()
    return float(tp + smooth) / float(tp + fn + smooth)


def specificity(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-7,
) -> float:
    """Compute specificity (true negative rate).

    Fraction of ground truth negatives correctly identified.

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        smooth: Smoothing constant.

    Returns:
        Specificity in [0, 1].
    """
    pred = prediction.astype(bool).ravel()
    tgt  = target.astype(bool).ravel()
    tn = np.logical_and(~pred, ~tgt).sum()
    fp = np.logical_and(pred, ~tgt).sum()
    return float(tn + smooth) / float(tn + fp + smooth)


def precision_score(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-7,
) -> float:
    """Compute precision (positive predictive value).

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        smooth: Smoothing constant.

    Returns:
        Precision in [0, 1].
    """
    pred = prediction.astype(bool).ravel()
    tgt  = target.astype(bool).ravel()
    tp = np.logical_and(pred, tgt).sum()
    fp = np.logical_and(pred, ~tgt).sum()
    return float(tp + smooth) / float(tp + fp + smooth)


def volume_similarity(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute volume similarity (VS).

    .. math::
        VS = 1 - \\frac{|V_P - V_T|}{V_P + V_T}

    Values near 1 indicate similar volumes. Note: VS does not capture
    spatial alignment; use Dice or HD95 for that.

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.

    Returns:
        Volume similarity in [0, 1].
    """
    vp = prediction.astype(bool).sum()
    vt = target.astype(bool).sum()
    denom = vp + vt
    if denom == 0:
        return 1.0
    return 1.0 - abs(vp - vt) / denom


def _surface_voxels(binary_mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels of a binary mask.

    Uses morphological erosion: surface = mask XOR eroded(mask).

    Args:
        binary_mask: Boolean 3-D (or N-D) array.

    Returns:
        Boolean array of surface voxels.
    """
    struct = generate_binary_structure(binary_mask.ndim, 1)
    eroded = binary_erosion(binary_mask, structure=struct, border_value=1)
    return np.logical_xor(binary_mask, eroded)


def hausdorff_distance_95(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Optional[Tuple[float, ...]] = None,
    percentile: float = 95.0,
) -> float:
    """Compute the 95th-percentile Hausdorff distance.

    The HD95 is more robust than the maximum HD because it excludes extreme
    outliers caused by segmentation artefacts. Lower is better.

    Args:
        prediction: Binary prediction array (any number of spatial dims).
        target: Binary ground truth array (same shape).
        voxel_spacing: Voxel size in mm per dimension, e.g. ``(1.0, 1.0, 1.0)``.
            If None, distances are in voxels.
        percentile: Percentile of surface distances to report.

    Returns:
        HD at ``percentile`` in voxels or mm.
        Returns ``np.nan`` if either mask is empty.
    """
    pred_bool = prediction.astype(bool)
    tgt_bool  = target.astype(bool)

    if not pred_bool.any() or not tgt_bool.any():
        return np.nan

    pred_surface = _surface_voxels(pred_bool)
    tgt_surface  = _surface_voxels(tgt_bool)

    pred_pts = np.argwhere(pred_surface).astype(float)
    tgt_pts  = np.argwhere(tgt_surface).astype(float)

    # Scale by voxel spacing
    if voxel_spacing is not None:
        spacing = np.array(voxel_spacing, dtype=float)
        pred_pts *= spacing
        tgt_pts  *= spacing

    # Directed Hausdorff distances
    hd_pt = directed_hausdorff(pred_pts, tgt_pts)[0]
    hd_tp = directed_hausdorff(tgt_pts, pred_pts)[0]

    # For percentile HD, compute all nearest-neighbour distances
    # Use a faster approach: for each surface point, find min dist to other surface
    from scipy.spatial import cKDTree
    tree_tgt  = cKDTree(tgt_pts)
    tree_pred = cKDTree(pred_pts)

    dists_pt = tree_tgt.query(pred_pts, k=1, workers=-1)[0]
    dists_tp = tree_pred.query(tgt_pts, k=1, workers=-1)[0]

    all_dists = np.concatenate([dists_pt, dists_tp])
    return float(np.percentile(all_dists, percentile))


def average_surface_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Optional[Tuple[float, ...]] = None,
) -> float:
    """Compute mean symmetric surface distance.

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        voxel_spacing: Voxel size in mm per dimension.

    Returns:
        Mean surface distance in voxels or mm. ``np.nan`` if either mask empty.
    """
    pred_bool = prediction.astype(bool)
    tgt_bool  = target.astype(bool)

    if not pred_bool.any() or not tgt_bool.any():
        return np.nan

    pred_surface = _surface_voxels(pred_bool)
    tgt_surface  = _surface_voxels(tgt_bool)

    pred_pts = np.argwhere(pred_surface).astype(float)
    tgt_pts  = np.argwhere(tgt_surface).astype(float)

    if voxel_spacing is not None:
        spacing = np.array(voxel_spacing, dtype=float)
        pred_pts *= spacing
        tgt_pts  *= spacing

    from scipy.spatial import cKDTree
    tree_tgt  = cKDTree(tgt_pts)
    tree_pred = cKDTree(pred_pts)

    dists_pt = tree_tgt.query(pred_pts, k=1, workers=-1)[0]
    dists_tp = tree_pred.query(tgt_pts, k=1, workers=-1)[0]

    return float(np.concatenate([dists_pt, dists_tp]).mean())


# ---------------------------------------------------------------------------
# Multi-class confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a multi-class confusion matrix.

    Args:
        prediction: Integer label array ``(*spatial)``.
        target: Integer ground truth array of the same shape.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)``.
        ``cm[i, j]`` is the number of voxels with true label ``i``
        predicted as ``j``.
    """
    pred_flat = prediction.ravel().astype(np.int64)
    tgt_flat  = target.ravel().astype(np.int64)

    valid = (tgt_flat >= 0) & (tgt_flat < num_classes) & \
            (pred_flat >= 0) & (pred_flat < num_classes)

    cm = np.bincount(
        num_classes * tgt_flat[valid] + pred_flat[valid],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm


# ---------------------------------------------------------------------------
# SegmentationMetrics class (high-level API)
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """Compute the full metric suite for multi-class segmentation.

    Handles the BraTS convention of computing metrics for three
    *compound* regions:
        - **Whole Tumour (WT)**: all tumour labels (1 + 2 + 3)
        - **Tumour Core (TC)**: labels 1 + 3 (NCR + ET)
        - **Enhancing Tumour (ET)**: label 3 only

    Args:
        num_classes: Total number of classes (including background).
        voxel_spacing: Voxel spacing in mm ``(dz, dy, dx)``. Used for HD95
            in physical units.
        include_background: Whether class 0 is included in per-class metrics.
        compute_hd95: If False, skip the expensive HD95 computation.

    Example::

        metrics = SegmentationMetrics(num_classes=4, voxel_spacing=(1, 1, 1))
        scores = metrics.compute_all(prediction, ground_truth)
        print(scores["mean_dice"])
        print(scores["hausdorff_95"])
    """

    # BraTS region definitions
    BRATS_REGIONS = {
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": [3],
    }

    def __init__(
        self,
        num_classes: int = 4,
        voxel_spacing: Optional[Tuple[float, float, float]] = None,
        include_background: bool = False,
        compute_hd95: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.voxel_spacing = voxel_spacing
        self.include_background = include_background
        self.compute_hd95 = compute_hd95

        self._foreground_classes = (
            list(range(num_classes)) if include_background
            else list(range(1, num_classes))
        )

    def compute_all(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all metrics for a single prediction/target pair.

        Args:
            prediction: Integer label array ``(*spatial)``.
            target: Integer label array ``(*spatial)``.

        Returns:
            Dict with keys: ``mean_dice``, ``class_dice``, ``iou``,
            ``sensitivity``, ``specificity``, ``precision``, ``hausdorff_95``,
            ``volume_similarity``, plus BraTS region keys (``wt_dice``, etc.)
            if num_classes == 4.
        """
        results: Dict[str, float] = {}
        class_dice: Dict[int, float] = {}
        class_iou: Dict[int, float] = {}

        for c in self._foreground_classes:
            pred_c = (prediction == c)
            tgt_c  = (target == c)
            class_dice[c] = dice_coefficient(pred_c, tgt_c)
            class_iou[c]  = iou_score(pred_c, tgt_c)

        results["class_dice"] = class_dice
        results["class_iou"]  = class_iou
        results["mean_dice"]  = float(np.mean(list(class_dice.values())))
        results["mean_iou"]   = float(np.mean(list(class_iou.values())))

        # Per-class sensitivity, specificity, precision (averaged over classes)
        sens_list, spec_list, prec_list = [], [], []
        for c in self._foreground_classes:
            pred_c = (prediction == c)
            tgt_c  = (target == c)
            sens_list.append(sensitivity(pred_c, tgt_c))
            spec_list.append(specificity(pred_c, tgt_c))
            prec_list.append(precision_score(pred_c, tgt_c))
        results["sensitivity"] = float(np.mean(sens_list))
        results["specificity"] = float(np.mean(spec_list))
        results["precision"]   = float(np.mean(prec_list))

        # HD95 on whole foreground
        pred_fg = (prediction > 0)
        tgt_fg  = (target > 0)
        if self.compute_hd95:
            results["hausdorff_95"] = hausdorff_distance_95(
                pred_fg, tgt_fg, self.voxel_spacing
            )
            results["avg_surface_dist"] = average_surface_distance(
                pred_fg, tgt_fg, self.voxel_spacing
            )
        else:
            results["hausdorff_95"]    = np.nan
            results["avg_surface_dist"] = np.nan

        results["volume_similarity"] = volume_similarity(pred_fg, tgt_fg)

        # BraTS compound regions
        if self.num_classes == 4:
            for region, labels in self.BRATS_REGIONS.items():
                pred_r = np.isin(prediction, labels)
                tgt_r  = np.isin(target, labels)
                results[f"{region.lower()}_dice"] = dice_coefficient(pred_r, tgt_r)
                if self.compute_hd95:
                    results[f"{region.lower()}_hd95"] = hausdorff_distance_95(
                        pred_r, tgt_r, self.voxel_spacing
                    )

        return results

    def compute_batch(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """Compute mean metrics over a batch.

        Args:
            predictions: ``(B, *spatial)`` integer predictions.
            targets: ``(B, *spatial)`` integer targets.

        Returns:
            Dict of mean metric values across the batch.
        """
        all_results: List[Dict] = []
        for pred, tgt in zip(predictions, targets):
            r = self.compute_all(pred, tgt)
            all_results.append(r)

        # Aggregate
        aggregated: Dict[str, float] = {}
        scalar_keys = [k for k in all_results[0] if not isinstance(all_results[0][k], dict)]
        for key in scalar_keys:
            vals = [r[key] for r in all_results if not np.isnan(r.get(key, np.nan))]
            aggregated[key] = float(np.mean(vals)) if vals else np.nan

        # Aggregate per-class dicts
        if "class_dice" in all_results[0]:
            class_dicts = [r["class_dice"] for r in all_results]
            aggregated["class_dice"] = {
                c: float(np.mean([d[c] for d in class_dicts]))
                for c in class_dicts[0]
            }
        return aggregated

    def generate_report(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        case_ids: Optional[List[str]] = None,
    ) -> str:
        """Generate a formatted text report of evaluation results.

        Args:
            predictions: ``(N, *spatial)`` predictions.
            targets: ``(N, *spatial)`` ground truth.
            case_ids: Optional list of case identifiers.

        Returns:
            Formatted multi-line string report.
        """
        per_case_results = []
        for i, (pred, tgt) in enumerate(zip(predictions, targets)):
            r = self.compute_all(pred, tgt)
            case_id = case_ids[i] if case_ids else f"case_{i:04d}"
            per_case_results.append((case_id, r))

        # Summary stats
        all_dice = [r["mean_dice"] for _, r in per_case_results]
        all_hd95 = [r["hausdorff_95"] for _, r in per_case_results
                    if not np.isnan(r["hausdorff_95"])]

        lines = [
            "=" * 70,
            "  SEGMENTATION EVALUATION REPORT",
            "=" * 70,
            f"  Cases evaluated: {len(per_case_results)}",
            "",
            "  SUMMARY",
            "-" * 70,
            f"  Mean Dice:          {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}",
            f"  Median Dice:        {np.median(all_dice):.4f}",
        ]

        if all_hd95:
            lines.append(
                f"  Mean HD95:          {np.mean(all_hd95):.2f} ± {np.std(all_hd95):.2f}"
            )

        if self.num_classes == 4:
            for region in ("wt", "tc", "et"):
                vals = [r[f"{region}_dice"] for _, r in per_case_results]
                lines.append(
                    f"  {region.upper()} Dice:            "
                    f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                )

        lines += [
            "",
            "  PER-CASE RESULTS (top 5 best / worst)",
            "-" * 70,
        ]

        sorted_cases = sorted(per_case_results, key=lambda x: x[1]["mean_dice"], reverse=True)
        for label, cases in [("Best", sorted_cases[:5]), ("Worst", sorted_cases[-5:])]:
            lines.append(f"  {label}:")
            for case_id, r in cases:
                lines.append(
                    f"    {case_id:<30s}  Dice={r['mean_dice']:.4f}"
                    f"  HD95={r.get('hausdorff_95', float('nan')):.1f}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

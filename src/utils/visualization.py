"""Visualisation utilities for medical image segmentation.

Provides:
    - Segmentation mask overlay on MRI images (clinical-style)
    - Training curve plots (loss, Dice, LR)
    - Metric comparison bar charts across models
    - Attention gate map visualisation
    - Grad-CAM overlay visualisation
    - Uncertainty map visualisation
    - Multi-panel prediction figure (input | prediction | ground truth | overlay)

All plotting functions return ``matplotlib.figure.Figure`` objects so they can
be saved or embedded in notebooks / TensorBoard without side effects.

Colour conventions follow standard brain tumour segmentation:
    - NCR/NET (class 1): Red
    - Edema (class 2): Green
    - Enhancing tumour (class 3): Blue
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for server environments
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# BraTS segmentation colour map (RGBA, 0-1 range)
BRATS_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),   # Background: transparent
    1: (0.9, 0.2, 0.2, 0.7),   # NCR/NET: red
    2: (0.2, 0.8, 0.2, 0.7),   # Edema: green
    3: (0.2, 0.4, 0.9, 0.7),   # ET: blue
}

BRATS_LABELS = {
    1: "NCR/NET",
    2: "Edema",
    3: "Enhancing Tumour",
}


def _build_label_cmap(color_dict: Dict[int, Tuple]) -> np.ndarray:
    """Build an RGBA colour lookup table from a class→colour dict."""
    n_classes = max(color_dict.keys()) + 1
    cmap = np.zeros((n_classes, 4), dtype=np.float32)
    for cls, color in color_dict.items():
        cmap[cls] = color
    return cmap


# ---------------------------------------------------------------------------
# Core overlay
# ---------------------------------------------------------------------------

def overlay_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color_dict: Optional[Dict[int, Tuple]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Blend a segmentation mask onto a grayscale image.

    Args:
        image: 2-D grayscale image array ``(H, W)``, any float range.
        mask: 2-D integer label array ``(H, W)``.
        alpha: Opacity of the overlay (0 = transparent, 1 = opaque).
        color_dict: Mapping from class index to RGBA colour tuple (0-1 range).
            Defaults to ``BRATS_COLORS``.
        vmin: Minimum value for image normalisation. Auto if None.
        vmax: Maximum value for image normalisation. Auto if None.

    Returns:
        RGBA image array ``(H, W, 4)`` with overlay, dtype float32.
    """
    if color_dict is None:
        color_dict = BRATS_COLORS

    # Normalise image to [0, 1]
    lo = image.min() if vmin is None else vmin
    hi = image.max() if vmax is None else vmax
    img_norm = np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)

    # Grayscale to RGB
    img_rgb = np.stack([img_norm] * 3, axis=-1)  # (H, W, 3)

    # Build overlay
    overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    for cls, color in color_dict.items():
        class_mask = (mask == cls)
        if class_mask.any():
            overlay[class_mask] = color

    # Alpha composite: output = src_alpha * src + (1 - src_alpha) * dst
    out = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    out[..., :3] = (
        overlay[..., 3:4] * alpha * overlay[..., :3]
        + (1 - overlay[..., 3:4] * alpha) * img_rgb
    )
    out[..., 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Prediction figure
# ---------------------------------------------------------------------------

def create_prediction_figure(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    modality_names: Optional[List[str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Create a multi-panel comparison figure for a single case.

    Panels:
        1. Input modality (or first if multi-channel)
        2. Ground truth overlay (if provided)
        3. Prediction overlay
        4. Error map (prediction ≠ ground truth, if GT provided)

    Args:
        image: ``(C, H, W)`` or ``(H, W)`` image.
        prediction: ``(H, W)`` integer prediction.
        ground_truth: ``(H, W)`` integer ground truth (optional).
        modality_names: Names for multi-modal channels.
        title: Figure title string.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure.
    """
    if image.ndim == 3:
        # Use FLAIR (index 3) or T1ce (index 1) as display modality
        display_ch = min(image.shape[0] - 1, 3)
        display_img = image[display_ch]
        ch_name = (modality_names[display_ch] if modality_names else f"Ch {display_ch}")
    else:
        display_img = image
        ch_name = "Image"

    has_gt = ground_truth is not None
    n_panels = 4 if has_gt else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    # Panel 1: Input
    axes[0].imshow(display_img, cmap="gray", aspect="equal")
    axes[0].set_title(ch_name, fontsize=11)
    axes[0].axis("off")

    # Panel 2: Ground truth
    if has_gt:
        overlay_gt = overlay_segmentation(display_img, ground_truth)
        axes[1].imshow(overlay_gt, aspect="equal")
        axes[1].set_title("Ground Truth", fontsize=11)
        axes[1].axis("off")
        pred_ax = axes[2]
    else:
        pred_ax = axes[1]

    # Prediction panel
    overlay_pred = overlay_segmentation(display_img, prediction)
    pred_ax.imshow(overlay_pred, aspect="equal")
    pred_ax.set_title("Prediction", fontsize=11)
    pred_ax.axis("off")

    # Error map
    if has_gt:
        error_map = (prediction != ground_truth).astype(np.float32)
        axes[3].imshow(display_img, cmap="gray", aspect="equal", alpha=0.7)
        axes[3].imshow(error_map, cmap="Reds", alpha=0.5, aspect="equal",
                       vmin=0, vmax=1)
        axes[3].set_title("Error Map", fontsize=11)
        axes[3].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=BRATS_COLORS[c][:3], label=BRATS_LABELS[c])
        for c in sorted(BRATS_LABELS.keys())
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Union[Dict[str, List[float]], str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """Plot training / validation loss and Dice curves.

    Args:
        history: Dict with keys ``'loss'``, ``'val_loss'``, ``'mean_dice'``,
            ``'lr'``, each containing a list of values per epoch.
            Alternatively, a path to a CSV file saved by the Trainer.
        save_path: If provided, save the figure here.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(history, str):
        import csv
        history_dict: Dict[str, List[float]] = {}
        with open(history) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    try:
                        history_dict.setdefault(k, []).append(float(v))
                    except (ValueError, TypeError):
                        pass
        history = history_dict

    epochs = list(range(len(history.get("loss", []))))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    if "loss" in history:
        ax.plot(epochs, history["loss"], label="Train Loss", color="#2196F3", linewidth=2)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="#F44336",
                linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Dice
    ax = axes[1]
    for key, label, color in [
        ("mean_dice", "Mean Dice", "#4CAF50"),
        ("wt_dice",   "WT Dice",   "#2196F3"),
        ("tc_dice",   "TC Dice",   "#FF9800"),
        ("et_dice",   "ET Dice",   "#9C27B0"),
    ]:
        if key in history:
            ax.plot(epochs[:len(history[key])], history[key],
                    label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_title("Validation Dice Scores")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Learning rate
    ax = axes[2]
    if "lr" in history:
        ax.semilogy(epochs[:len(history["lr"])], history["lr"],
                    color="#607D8B", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved training curves to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Metric comparison chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    model_results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing metrics across multiple models.

    Args:
        model_results: Dict mapping model name → metric dict.
            Example: ``{"UNet": {"mean_dice": 0.85}, "Attention UNet": ...}``
        metrics: Metrics to include. Defaults to Dice scores.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    if metrics is None:
        metrics = ["mean_dice", "wt_dice", "tc_dice", "et_dice"]

    model_names = list(model_results.keys())
    n_models  = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))  # type: ignore

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, results) in enumerate(model_results.items()):
        values = [results.get(m, 0.0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x + width * (n_models - 1) / 2)
    metric_labels = [m.replace("_", " ").title() for m in metrics]
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved metric comparison to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Attention maps
# ---------------------------------------------------------------------------

def plot_attention_maps(
    image: np.ndarray,
    attention_maps: List[np.ndarray],
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualise attention gate maps alongside the input image.

    Args:
        image: 2-D grayscale image ``(H, W)``.
        attention_maps: List of attention map arrays, each ``(H', W')``.
        titles: Subplot titles.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    n = len(attention_maps) + 1  # +1 for input
    if figsize is None:
        figsize = (4 * n, 4)

    if titles is None:
        titles = [f"Level {i}" for i in range(len(attention_maps))]

    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.suptitle("Attention Gate Maps", fontsize=13, fontweight="bold")

    axes[0].imshow(image, cmap="gray", aspect="equal")
    axes[0].set_title("Input", fontsize=11)
    axes[0].axis("off")

    for i, (amap, title) in enumerate(zip(attention_maps, titles)):
        # Upsample if needed
        if amap.shape != image.shape:
            from scipy.ndimage import zoom
            factors = [s / a for s, a in zip(image.shape, amap.shape)]
            amap = zoom(amap, factors, order=1)

        axes[i + 1].imshow(image, cmap="gray", aspect="equal", alpha=0.6)
        im = axes[i + 1].imshow(amap, cmap="hot", aspect="equal", alpha=0.7,
                                 vmin=0, vmax=1)
        axes[i + 1].set_title(title, fontsize=11)
        axes[i + 1].axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Attention Weight")

    plt.tight_layout(rect=[0, 0, 0.91, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved attention maps to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Grad-CAM overlay
# ---------------------------------------------------------------------------

def plot_gradcam(
    image: np.ndarray,
    cam: np.ndarray,
    mask: Optional[np.ndarray] = None,
    class_name: str = "Target Class",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualise a Grad-CAM heatmap overlaid on the input image.

    Args:
        image: 2-D grayscale image ``(H, W)``.
        cam: Grad-CAM map ``(H, W)``, values in [0, 1].
        mask: Optional ground truth mask for reference.
        class_name: Name of the target class.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    n_panels = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle(f"Grad-CAM: {class_name}", fontsize=13, fontweight="bold")

    # Input
    axes[0].imshow(image, cmap="gray", aspect="equal")
    axes[0].set_title("Input", fontsize=11)
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(image, cmap="gray", aspect="equal", alpha=0.6)
    im = axes[1].imshow(cam, cmap="jet", alpha=0.5, aspect="equal", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Ground truth mask
    if mask is not None:
        axes[2].imshow(
            overlay_segmentation(image, mask), aspect="equal"
        )
        axes[2].set_title("Ground Truth", fontsize=11)
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved Grad-CAM figure to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Uncertainty visualisation
# ---------------------------------------------------------------------------

def plot_uncertainty(
    image: np.ndarray,
    mean_prediction: np.ndarray,
    uncertainty_map: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    uncertainty_type: str = "Predictive Entropy",
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualise MC Dropout uncertainty alongside the mean prediction.

    Args:
        image: 2-D grayscale image ``(H, W)``.
        mean_prediction: Integer label array ``(H, W)``.
        uncertainty_map: Uncertainty array ``(H, W)``, e.g. entropy.
        ground_truth: Optional ground truth mask ``(H, W)``.
        uncertainty_type: Label for the uncertainty measure.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    has_gt = ground_truth is not None
    n_panels = 4 if has_gt else 3
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle("Prediction with Uncertainty Estimation", fontsize=13, fontweight="bold")

    # Input
    axes[0].imshow(image, cmap="gray", aspect="equal")
    axes[0].set_title("Input", fontsize=11)
    axes[0].axis("off")

    # Mean prediction
    axes[1].imshow(overlay_segmentation(image, mean_prediction), aspect="equal")
    axes[1].set_title("Mean Prediction", fontsize=11)
    axes[1].axis("off")

    # Uncertainty
    im_u = axes[2].imshow(uncertainty_map, cmap="plasma", aspect="equal")
    axes[2].set_title(f"Uncertainty\n({uncertainty_type})", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im_u, ax=axes[2], fraction=0.046, pad=0.04)

    # Ground truth
    if has_gt:
        axes[3].imshow(overlay_segmentation(image, ground_truth), aspect="equal")
        axes[3].set_title("Ground Truth", fontsize=11)
        axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved uncertainty figure to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Convenience: save figure
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: Union[str, Path], dpi: int = 150) -> None:
    """Save a matplotlib figure with standard settings.

    Args:
        fig: Figure to save.
        path: Output file path (extension determines format: png, pdf, svg).
        dpi: Resolution in dots per inch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure: %s", path)

"""Model explainability tools for medical image segmentation.

Implements:
    - Grad-CAM adapted for dense segmentation (class-activation maps)
    - Attention map extraction and visualisation from Attention U-Net
    - MC Dropout uncertainty estimation
    - Ensemble-based uncertainty (if multiple checkpoints provided)

These tools are critical for FDA GMLP compliance: they provide radiologists
with visual evidence supporting the model's predictions, enabling meaningful
human oversight before clinical decisions.

References:
    Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-Based Localization. ICCV.

    Gal & Ghahramani (2016). Dropout as a Bayesian Approximation.
    ICML.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM for Segmentation
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for segmentation models.

    For segmentation, we compute Grad-CAM with respect to a target class
    averaged over the spatial output (or a specified ROI mask).

    Standard classification Grad-CAM:
        1. Forward pass → get activations at target layer.
        2. Backward pass with respect to target class score.
        3. Average gradients over spatial dims → channel weights α_k.
        4. CAM = ReLU(Σ α_k * A_k), upsample to input resolution.

    For segmentation:
        - The class score is the sum of logits for the target class
          within an optional ROI mask.

    Args:
        model: PyTorch model.
        target_layer: Module reference to the layer to visualise.
            Typically the last encoder or bottleneck conv block.

    Example::

        model = AttentionUNet(in_channels=4, num_classes=4)
        # Target the bottleneck layer
        gradcam = GradCAM(model, target_layer=model.bottleneck.conv2)
        image = torch.randn(1, 4, 240, 240)
        cam = gradcam.generate(image, target_class=3)  # ET class
        # cam.shape == (1, 1, 240, 240), values in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_backward_hook(backward_hook)
        )

    def remove_hooks(self) -> None:
        """Remove registered hooks (call when done to avoid memory leaks)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        roi_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor ``(1, C, H, W)`` or
                ``(1, C, D, H, W)``. Must have batch size 1.
            target_class: Class index to visualise.
            roi_mask: Optional spatial mask ``(1, 1, H, W)`` restricting
                the region used for gradient averaging.

        Returns:
            Grad-CAM heatmap as numpy array, shape ``(H, W)`` or ``(D, H, W)``,
            normalised to [0, 1].
        """
        if input_tensor.shape[0] != 1:
            raise ValueError("GradCAM requires batch size 1.")

        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(False)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(input_tensor)  # (1, C, *spatial)

        # Compute class score: sum of target-class logits (optionally masked)
        class_logits = logits[:, target_class]  # (1, *spatial)
        if roi_mask is not None:
            class_logits = class_logits * roi_mask.squeeze(1)
        score = class_logits.sum()

        # Backward pass
        score.backward()

        # Compute weights: global average pooling over spatial dims
        # self._gradients: (1, C_layer, *spatial)
        grads = self._gradients  # (1, C_layer, *spatial)
        acts  = self._activations  # (1, C_layer, *spatial)

        # Pool gradients over all spatial dims
        spatial_dims = tuple(range(2, grads.ndim))
        weights = grads.mean(dim=spatial_dims, keepdim=True)  # (1, C_layer, 1, ...)

        # Weighted combination of feature maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, *spatial)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, *([1] * (cam.ndim - 2)))
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, *([1] * (cam.ndim - 2)))
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Upsample to input resolution
        spatial_size = input_tensor.shape[2:]
        cam_up = F.interpolate(
            cam,
            size=spatial_size,
            mode="bilinear" if cam.ndim == 4 else "trilinear",
            align_corners=False,
        )

        return cam_up.squeeze().detach().cpu().numpy()

    def generate_multi_class(
        self,
        input_tensor: torch.Tensor,
        classes: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Generate Grad-CAM maps for multiple target classes.

        Args:
            input_tensor: Input tensor ``(1, C, H, W)``.
            classes: List of class indices. Defaults to all classes.

        Returns:
            Dict mapping class index → Grad-CAM heatmap.
        """
        if classes is None:
            logits = self.model(input_tensor.requires_grad_(False))
            classes = list(range(logits.shape[1]))

        cams = {}
        for cls in classes:
            cams[cls] = self.generate(input_tensor, cls)
        return cams

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Attention Map Visualiser (Attention U-Net)
# ---------------------------------------------------------------------------

class AttentionMapVisualizer:
    """Extract and visualise attention gate maps from Attention U-Net.

    Attention maps show which spatial regions the model focuses on at
    each decoder level, providing interpretability without any backward pass.

    Args:
        model: AttentionUNet instance. Maps are read from
            ``model.attention_maps`` after a forward pass.

    Example::

        model = AttentionUNet(in_channels=4, num_classes=4)
        visualizer = AttentionMapVisualizer(model)
        image = torch.randn(1, 4, 240, 240)
        with torch.no_grad():
            logits = model(image)
        maps = visualizer.get_maps()
        # maps[0].shape == (1, 1, 240, 240) at full resolution
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def get_maps(self, upsample: bool = True) -> List[np.ndarray]:
        """Return attention maps from the last forward pass.

        Args:
            upsample: If True, upsample all maps to the input resolution.

        Returns:
            List of attention maps (numpy arrays), one per decoder level.
            Each map shape: ``(H, W)`` for 2-D models.
        """
        if not hasattr(self.model, "attention_maps"):
            raise AttributeError(
                "Model does not have attention_maps attribute. "
                "Use AttentionUNet instead of UNet."
            )

        maps = self.model.attention_maps  # list of (1, 1, *spatial) tensors

        if not maps:
            raise RuntimeError("No forward pass has been performed yet.")

        result = []
        for alpha in maps:
            arr = alpha.squeeze().cpu().numpy()  # (H, W) or (D, H, W)
            # Normalise to [0, 1]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            result.append(arr)

        return result

    def aggregate_map(self, upsample_size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Aggregate attention maps across all levels by averaging.

        Args:
            upsample_size: Target spatial size for upsampling before averaging.

        Returns:
            Aggregated attention map, normalised to [0, 1].
        """
        maps = self.get_maps()

        if upsample_size is not None:
            upsampled = []
            for m in maps:
                t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float()
                mode = "bilinear" if t.ndim == 4 else "trilinear"
                t = F.interpolate(t, size=upsample_size, mode=mode, align_corners=False)
                upsampled.append(t.squeeze().numpy())
            maps = upsampled

        aggregated = np.mean(np.stack(maps, axis=0), axis=0)
        return (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min() + 1e-8)


# ---------------------------------------------------------------------------
# MC Dropout Uncertainty Estimation
# ---------------------------------------------------------------------------

class UncertaintyEstimator:
    """Monte Carlo Dropout uncertainty estimator for segmentation models.

    Enables Bayesian approximation of prediction uncertainty by running
    multiple stochastic forward passes with dropout active.

    Two uncertainty measures are computed:
        - **Predictive entropy**: total uncertainty from all sources.
        - **Mutual information** (epistemic uncertainty): uncertainty
          due to limited training data / model capacity.

    Args:
        model: Model with dropout layers (built with ``dropout_p > 0``).
        n_passes: Number of Monte Carlo samples per prediction.

    Example::

        estimator = UncertaintyEstimator(model, n_passes=30)
        mean_pred, uncertainty = estimator.predict(image)
        # uncertainty["entropy"].shape == (1, H, W)
        # High values indicate uncertain regions → flag for radiologist review
    """

    def __init__(self, model: nn.Module, n_passes: int = 20) -> None:
        self.model = model
        self.n_passes = n_passes

    def predict(
        self,
        input_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        """Run MC Dropout inference.

        Args:
            input_tensor: Input tensor (any batch size).

        Returns:
            Tuple of:
                - Mean prediction tensor ``(B, C, *spatial)`` (softmax probs).
                - Uncertainty dict with keys ``'entropy'``, ``'variance'``,
                  ``'mutual_information'``, all shape ``(B, *spatial)``.
        """
        # Activate dropout layers
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

        preds = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                logits = self.model(input_tensor)
                preds.append(F.softmax(logits, dim=1))

        # Stack: (n_passes, B, C, *spatial)
        stacked = torch.stack(preds, dim=0)

        mean_pred = stacked.mean(dim=0)          # (B, C, *spatial)
        variance  = stacked.var(dim=0)           # (B, C, *spatial)

        # Predictive entropy: H[E[p(y|x)]]
        mean_np = mean_pred.cpu().numpy()
        entropy = -np.sum(
            mean_np * np.log(mean_np + 1e-8), axis=1
        )  # (B, *spatial)

        # Expected entropy: E[H[p(y|x,w)]]
        preds_np = stacked.cpu().numpy()
        exp_entropy = -np.mean(
            np.sum(preds_np * np.log(preds_np + 1e-8), axis=2), axis=0
        )  # (B, *spatial)

        # Mutual information (epistemic uncertainty) = predictive_entropy - expected_entropy
        mutual_info = entropy - exp_entropy

        uncertainty = {
            "entropy":            entropy,
            "variance":           variance.mean(dim=1).cpu().numpy(),
            "mutual_information": mutual_info,
        }

        # Reset model to eval mode
        self.model.eval()
        return mean_pred, uncertainty

    def get_uncertainty_mask(
        self,
        uncertainty_map: np.ndarray,
        threshold_percentile: float = 90.0,
    ) -> np.ndarray:
        """Generate a binary mask of high-uncertainty regions.

        Useful for flagging regions that require radiologist review.

        Args:
            uncertainty_map: Spatial uncertainty array ``(*spatial)``.
            threshold_percentile: Percentile threshold above which regions
                are flagged as uncertain.

        Returns:
            Boolean mask of high-uncertainty voxels.
        """
        threshold = np.percentile(uncertainty_map, threshold_percentile)
        return uncertainty_map > threshold

    def calibration_analysis(
        self,
        predictions: List[np.ndarray],
        targets: List[np.ndarray],
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Compute reliability diagram data for calibration assessment.

        A well-calibrated model should have confidence ≈ accuracy at all
        confidence levels (diagonal reliability diagram).

        Args:
            predictions: List of probability arrays ``(C, *spatial)``.
            targets: List of integer label arrays ``(*spatial)``.
            n_bins: Number of confidence bins.

        Returns:
            Dict with ``'bin_confidences'``, ``'bin_accuracies'``,
            ``'bin_counts'``, and ``'ece'`` (expected calibration error).
        """
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_sums = np.zeros(n_bins)
        bin_correct = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)

        for probs, tgt in zip(predictions, targets):
            # Max probability and predicted class
            max_probs = probs.max(axis=0).ravel()
            pred_class = probs.argmax(axis=0).ravel()
            tgt_flat = tgt.ravel()

            for b in range(n_bins):
                mask = (max_probs >= bin_edges[b]) & (max_probs < bin_edges[b + 1])
                if mask.sum() > 0:
                    bin_sums[b] += max_probs[mask].sum()
                    bin_correct[b] += (pred_class[mask] == tgt_flat[mask]).sum()
                    bin_counts[b] += mask.sum()

        valid = bin_counts > 0
        bin_confidences = np.where(valid, bin_sums / np.maximum(bin_counts, 1), 0.0)
        bin_accuracies  = np.where(valid, bin_correct / np.maximum(bin_counts, 1), 0.0)

        # Expected Calibration Error
        total = bin_counts.sum()
        ece = float(
            np.sum(bin_counts[valid] / total * np.abs(
                bin_confidences[valid] - bin_accuracies[valid]
            ))
        )

        return {
            "bin_confidences": bin_confidences,
            "bin_accuracies":  bin_accuracies,
            "bin_counts":      bin_counts,
            "ece":             ece,
        }

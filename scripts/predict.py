#!/usr/bin/env python3
"""Inference script for MedSeg Pipeline — single case prediction.

Accepts multi-modal NIfTI files (BraTS convention) or a DICOM series
directory and produces a segmentation mask in NIfTI format, along with
optional uncertainty maps and Grad-CAM visualisations.

Usage::

    # From NIfTI files (BraTS)
    python scripts/predict.py \\
        --checkpoint experiments/run1/checkpoints/best_model.pth \\
        --t1    /path/to/case/t1.nii.gz \\
        --t1ce  /path/to/case/t1ce.nii.gz \\
        --t2    /path/to/case/t2.nii.gz \\
        --flair /path/to/case/flair.nii.gz \\
        --output predictions/case001_seg.nii.gz \\
        --uncertainty \\
        --gradcam

    # From DICOM directory (single modality)
    python scripts/predict.py \\
        --checkpoint experiments/run1/checkpoints/best_model.pth \\
        --dicom_dir /path/to/dicom_series \\
        --output predictions/case001_seg.nii.gz
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import get_model
from data.dicom_loader import DICOMLoader, load_nifti, save_nifti
from data.preprocessing import IntensityNormalizer
from evaluation.explainability import GradCAM, UncertaintyEstimator
from utils.visualization import plot_gradcam, plot_uncertainty, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSeg Pipeline — Single Case Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output NIfTI path for segmentation mask.")

    # Input options (mutually exclusive)
    input_group = parser.add_argument_group("Input (NIfTI, BraTS convention)")
    input_group.add_argument("--t1",    type=str, default=None, help="T1 NIfTI path.")
    input_group.add_argument("--t1ce",  type=str, default=None, help="T1ce NIfTI path.")
    input_group.add_argument("--t2",    type=str, default=None, help="T2 NIfTI path.")
    input_group.add_argument("--flair", type=str, default=None, help="FLAIR NIfTI path.")

    dicom_group = parser.add_argument_group("Input (DICOM)")
    dicom_group.add_argument("--dicom_dir", type=str, default=None,
                             help="DICOM series directory (single modality).")

    # Inference options
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--uncertainty", action="store_true",
                        help="Compute and save MC Dropout uncertainty map.")
    parser.add_argument("--n_mc_passes", type=int, default=20,
                        help="Number of MC Dropout passes for uncertainty.")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM maps (saved alongside output).")
    parser.add_argument("--gradcam_class", type=int, default=3,
                        help="Target class for Grad-CAM (default: 3 = ET).")
    parser.add_argument("--tta", action="store_true",
                        help="Apply test-time augmentation (flips).")
    parser.add_argument("--slice_2d", action="store_true",
                        help="Run inference slice-by-slice (for 2-D models).")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    model = get_model(
        cfg.get("architecture", "attention_unet"),
        in_channels=cfg.get("in_channels", 4),
        num_classes=cfg.get("num_classes", 4),
        spatial_dims=cfg.get("spatial_dims", 2),
        features=tuple(cfg.get("features", [64, 128, 256, 512])),
        norm_type=cfg.get("norm_type", "batch"),
        dropout_p=cfg.get("dropout_p", 0.1),
        residual=cfg.get("residual", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("Loaded model: %s  (epoch %d)", cfg.get("architecture"), ckpt.get("epoch", 0))
    return model, cfg


def load_brats_volume(t1, t1ce, t2, flair) -> np.ndarray:
    """Load and normalise 4-channel BraTS input."""
    normaliser = IntensityNormalizer(method="zscore", mask_background=True)
    channels = []
    for path in (t1, t1ce, t2, flair):
        data, _, _ = load_nifti(path)
        channels.append(normaliser(data))
    volume = np.stack(channels, axis=0).astype(np.float32)  # (4, H, W, D)
    return volume


def load_dicom_volume(dicom_dir: str) -> np.ndarray:
    """Load and preprocess a DICOM series."""
    loader = DICOMLoader(dicom_dir)
    volume, metadata = loader.load()
    normaliser = IntensityNormalizer(method="zscore", mask_background=True)
    volume = normaliser(volume)
    return volume[np.newaxis].astype(np.float32)  # (1, D, H, W)


def predict_2d_slicewise(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """Run 2-D inference over all axial slices of a 3-D volume.

    Args:
        model: 2-D segmentation model.
        volume: ``(C, H, W, D)`` volume.
        device: Inference device.
        batch_size: Slices processed simultaneously.

    Returns:
        Prediction array ``(H, W, D)`` with integer class labels.
    """
    C, H, W, D = volume.shape
    prediction = np.zeros((H, W, D), dtype=np.int64)

    slices_range = range(D)
    n_batches = (D + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, D)

        batch_slices = volume[:, :, :, start:end]  # (C, H, W, batch)
        batch_tensor = torch.from_numpy(
            batch_slices.transpose(3, 0, 1, 2)  # (batch, C, H, W)
        ).float().to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            preds  = logits.argmax(dim=1).cpu().numpy()  # (batch, H, W)

        prediction[:, :, start:end] = preds.transpose(1, 2, 0)

    return prediction


def apply_tta(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply test-time augmentation (horizontal/vertical flips).

    Averages softmax predictions across original and flipped inputs.

    Args:
        model: Segmentation model.
        tensor: Input tensor ``(1, C, H, W)``.
        device: Device.

    Returns:
        Averaged softmax predictions ``(1, num_classes, H, W)``.
    """
    import torch.nn.functional as F
    augmented = [
        tensor,
        tensor.flip(dims=[-1]),           # horizontal flip
        tensor.flip(dims=[-2]),           # vertical flip
        tensor.flip(dims=[-1, -2]),       # both
    ]

    preds = []
    model.eval()
    with torch.no_grad():
        for aug in augmented:
            logits = model(aug.to(device))
            probs  = F.softmax(logits, dim=1)
            preds.append(probs)

    # Un-flip augmented predictions
    preds[1] = preds[1].flip(dims=[-1])
    preds[2] = preds[2].flip(dims=[-2])
    preds[3] = preds[3].flip(dims=[-1, -2])

    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", device)

    # Load model
    model, model_cfg = load_model_from_checkpoint(args.checkpoint, device)

    # Load input volume
    if args.dicom_dir:
        logger.info("Loading DICOM series from %s", args.dicom_dir)
        volume = load_dicom_volume(args.dicom_dir)
        affine = np.eye(4)
    elif args.t1 and args.t1ce and args.t2 and args.flair:
        logger.info("Loading BraTS NIfTI modalities...")
        volume = load_brats_volume(args.t1, args.t1ce, args.t2, args.flair)
        _, affine, _ = load_nifti(args.t1)  # use T1 affine for output
    else:
        logger.error("Provide either --dicom_dir OR all four NIfTI modalities (--t1, --t1ce, --t2, --flair).")
        sys.exit(1)

    logger.info("Input volume shape: %s", volume.shape)

    # Inference
    start_time = time.perf_counter()

    if args.slice_2d or model_cfg.get("spatial_dims", 2) == 2:
        logger.info("Running 2-D slice-wise inference...")
        # volume: (C, H, W, D)
        if volume.ndim == 4:
            prediction = predict_2d_slicewise(model, volume, device)
        else:
            raise ValueError("Expected 4-D volume (C, H, W, D) for 2-D inference.")
    else:
        # 3-D inference
        logger.info("Running 3-D patch-based inference...")
        # For simplicity: run on full volume (requires sufficient GPU memory)
        tensor = torch.from_numpy(volume).unsqueeze(0).float()
        if args.tta:
            logger.info("Applying test-time augmentation...")
            probs = apply_tta(model, tensor, device)
            prediction = probs.argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            with torch.no_grad():
                logits = model(tensor.to(device))
                prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    inference_time = time.perf_counter() - start_time
    logger.info("Inference completed in %.2f seconds.", inference_time)

    # Remap class 3 → label 4 (BraTS convention for ET)
    output_mask = prediction.copy()
    output_mask[prediction == 3] = 4

    # Save prediction
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(output_mask.astype(np.int16), affine, output_path)
    logger.info("Saved segmentation to %s", output_path)

    # Unique classes in prediction
    unique_classes = np.unique(output_mask)
    class_names = {0: "Background", 1: "NCR/NET", 2: "Edema", 4: "Enhancing Tumour"}
    logger.info(
        "Detected classes: %s",
        {c: class_names.get(c, f"Class {c}") for c in unique_classes if c != 0},
    )

    # Uncertainty estimation
    if args.uncertainty:
        logger.info("Computing MC Dropout uncertainty (%d passes)...", args.n_mc_passes)
        if not any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d))
                   for m in model.modules()):
            logger.warning("No dropout layers found. Uncertainty estimates will be zero.")

        estimator = UncertaintyEstimator(model, n_passes=args.n_mc_passes)
        if volume.ndim == 4:
            tensor = torch.from_numpy(volume[:, :, :, volume.shape[-1]//2]).unsqueeze(0).float()
        else:
            tensor = torch.from_numpy(volume).unsqueeze(0).float()

        mean_pred, uncertainty_maps = estimator.predict(tensor.to(device))

        entropy_map = uncertainty_maps["entropy"].squeeze()
        entropy_path = output_path.parent / (output_path.stem.replace(".nii", "") + "_uncertainty.nii.gz")
        save_nifti(entropy_map.astype(np.float32), affine[:2, :2] if affine.shape[0] > 2 else np.eye(3),
                   entropy_path)

        # Quick summary
        high_unc_mask = estimator.get_uncertainty_mask(entropy_map, threshold_percentile=90)
        logger.info(
            "Uncertainty: mean entropy=%.4f  high-uncertainty voxels=%.1f%%",
            entropy_map.mean(),
            100.0 * high_unc_mask.sum() / entropy_map.size,
        )
        logger.info("Uncertainty map saved to %s", entropy_path)

    # Grad-CAM
    if args.gradcam:
        logger.info("Generating Grad-CAM for class %d...", args.gradcam_class)
        try:
            # Target the last encoder conv block (heuristic)
            if hasattr(model, "bottleneck"):
                target_layer = model.bottleneck.conv2
            elif hasattr(model, "encoders"):
                target_layer = model.encoders[-1].conv_block.conv2
            else:
                logger.warning("Cannot find suitable target layer for Grad-CAM.")
                target_layer = None

            if target_layer is not None:
                # Use middle axial slice for 3-D volumes
                if volume.ndim == 4:
                    mid = volume.shape[-1] // 2
                    slice_tensor = torch.from_numpy(volume[:, :, :, mid]).unsqueeze(0).float().to(device)
                else:
                    slice_tensor = torch.from_numpy(volume).unsqueeze(0).float().to(device)

                with GradCAM(model, target_layer) as gcam:
                    cam = gcam.generate(slice_tensor, target_class=args.gradcam_class)

                cam_path = output_path.parent / (output_path.stem.replace(".nii", "") + f"_gradcam_cls{args.gradcam_class}.png")
                display_img = volume[min(3, volume.shape[0]-1), :, :, volume.shape[-1]//2] if volume.ndim == 4 else volume.squeeze()
                fig = plot_gradcam(
                    image=display_img,
                    cam=cam,
                    class_name=f"Class {args.gradcam_class}",
                )
                save_figure(fig, cam_path)
                logger.info("Grad-CAM saved to %s", cam_path)
        except Exception as exc:
            logger.warning("Grad-CAM failed: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()

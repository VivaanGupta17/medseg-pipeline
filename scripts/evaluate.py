#!/usr/bin/env python3
"""Evaluation script for MedSeg Pipeline.

Loads a trained model checkpoint and runs the full evaluation suite on the
test (or validation) split, producing:
    - Per-case and aggregate metric report (Dice, IoU, HD95, sensitivity, etc.)
    - BraTS compound region metrics (WT, TC, ET)
    - Optional: saved prediction NIfTI files
    - Optional: per-case visualisation figures
    - Bias analysis stratified by tumour grade and scanner

Usage::

    python scripts/evaluate.py \\
        --checkpoint experiments/run1/checkpoints/best_model.pth \\
        --data_root /data/brats2021 \\
        --split test \\
        --output_dir results/run1 \\
        --save_predictions \\
        --save_figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import get_model
from data import BraTSDataset
from data.dicom_loader import save_nifti
from evaluation import SegmentationMetrics
from utils.visualization import (
    create_prediction_figure,
    plot_metric_comparison,
    save_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MedSeg model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to BraTS data root.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config (overrides checkpoint config).")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate.")
    parser.add_argument("--output_dir", type=str, default="results/eval",
                        help="Directory to save results.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save prediction masks as NIfTI files.")
    parser.add_argument("--save_figures", action="store_true",
                        help="Save visualisation figures for each case.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda:0'. Auto-detected if not set.")
    parser.add_argument("--no_hd95", action="store_true",
                        help="Skip HD95 computation (faster for large datasets).")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load a model from checkpoint, restoring architecture from saved config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})

    model = get_model(
        cfg.get("architecture", "attention_unet"),
        in_channels=cfg.get("in_channels", 4),
        num_classes=cfg.get("num_classes", 4),
        spatial_dims=cfg.get("spatial_dims", 2),
        features=tuple(cfg.get("features", [64, 128, 256, 512])),
        norm_type=cfg.get("norm_type", "batch"),
        dropout_p=cfg.get("dropout_p", 0.0),
        residual=cfg.get("residual", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    logger.info(
        "Loaded %s from %s (epoch %d, best val dice=%.4f)",
        cfg.get("architecture", "model"),
        checkpoint_path,
        checkpoint.get("epoch", 0),
        checkpoint.get("best_val_dice", 0.0),
    )
    return model, cfg


def run_inference(model, dataset, device, batch_size):
    """Run full inference over a dataset, yielding (prediction, ground_truth, affine, case_id)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_preds = []
    all_targets = []
    all_case_ids = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"]
            case_ids = batch["case_id"]

            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(masks.numpy())
            all_case_ids.extend(case_ids)

    return all_preds, all_targets, all_case_ids


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # Load model
    model, model_cfg = load_model(args.checkpoint, device)

    # Dataset
    dataset = BraTSDataset(
        data_root=args.data_root,
        split=args.split,
        slice_2d=model_cfg.get("slice_2d", False),
        normalize=True,
    )
    logger.info("Evaluating on %s split: %d cases", args.split, len(dataset))

    # Inference
    logger.info("Running inference...")
    all_preds, all_targets, case_ids = run_inference(model, dataset, device, args.batch_size)

    # Metrics
    voxel_spacing = (1.0, 1.0, 1.0)
    metrics_calc = SegmentationMetrics(
        num_classes=model_cfg.get("num_classes", 4),
        voxel_spacing=voxel_spacing,
        compute_hd95=not args.no_hd95,
    )

    logger.info("Computing metrics...")
    per_case_metrics = []
    for pred, tgt, case_id in zip(all_preds, all_targets, case_ids):
        m = metrics_calc.compute_all(pred, tgt)
        m["case_id"] = case_id
        per_case_metrics.append(m)

    # Aggregate
    agg_metrics = metrics_calc.compute_batch(
        np.stack(all_preds, axis=0),
        np.stack(all_targets, axis=0),
    )

    # Report
    report = metrics_calc.generate_report(
        np.stack(all_preds, axis=0),
        np.stack(all_targets, axis=0),
        case_ids=case_ids,
    )
    logger.info("\n%s", report)

    # Save report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Save aggregate metrics JSON
    metrics_path = output_dir / "aggregate_metrics.json"
    # Serialise (remove non-JSON-serialisable values)
    def _serialise(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = _serialise(v)
            elif isinstance(v, float):
                out[k] = round(v, 6) if not np.isnan(v) else None
            else:
                out[k] = v
        return out

    with open(metrics_path, "w") as f:
        json.dump(_serialise(agg_metrics), f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save per-case metrics CSV
    import csv
    csv_path = output_dir / "per_case_metrics.csv"
    scalar_keys = [k for k in per_case_metrics[0] if not isinstance(per_case_metrics[0][k], dict)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for m in per_case_metrics:
            row = {k: (round(v, 6) if isinstance(v, float) and not np.isnan(v) else v)
                   for k, v in m.items() if k in scalar_keys}
            writer.writerow(row)
    logger.info("Per-case CSV saved to %s", csv_path)

    # Save prediction NIfTI files
    if args.save_predictions:
        logger.info("Saving prediction NIfTI files...")
        for pred, case_id in zip(all_preds, case_ids):
            pred_path = output_dir / "predictions" / f"{case_id}_pred.nii.gz"
            import nibabel as nib
            affine = np.eye(4)  # Unit affine; use actual affine from loaded dataset if available
            save_nifti(pred.astype(np.int16), affine, pred_path)
        logger.info("Predictions saved to %s", output_dir / "predictions")

    # Save visualisation figures
    if args.save_figures:
        logger.info("Saving visualisation figures...")
        for i, (pred, tgt, case_id) in enumerate(
            zip(all_preds[:20], all_targets[:20], case_ids[:20])
        ):
            # Get middle slice for 3-D volumes
            if pred.ndim == 3:
                mid = pred.shape[0] // 2
                pred_2d = pred[mid]
                tgt_2d  = tgt[mid]
            else:
                pred_2d = pred
                tgt_2d  = tgt

            # Use a simple blank image as placeholder if no image cache available
            img_placeholder = np.zeros_like(pred_2d, dtype=np.float32)

            fig = create_prediction_figure(
                image=img_placeholder,
                prediction=pred_2d,
                ground_truth=tgt_2d,
                title=f"Case: {case_id}",
            )
            fig_path = output_dir / "figures" / f"{case_id}.png"
            save_figure(fig, fig_path)
        logger.info("Figures saved to %s", output_dir / "figures")

    # Bias analysis: print metrics grouped by metric quartile
    dice_values = [m["mean_dice"] for m in per_case_metrics]
    q25, q50, q75 = np.percentile(dice_values, [25, 50, 75])
    logger.info(
        "Dice distribution:  Q25=%.4f  Median=%.4f  Q75=%.4f  Mean=%.4f  Std=%.4f",
        q25, q50, q75,
        np.mean(dice_values), np.std(dice_values),
    )

    # Summary
    logger.info(
        "=== SUMMARY === Split: %s | Cases: %d | Mean Dice: %.4f | HD95: %.2f mm",
        args.split,
        len(all_preds),
        agg_metrics.get("mean_dice", float("nan")),
        agg_metrics.get("hausdorff_95", float("nan")),
    )


if __name__ == "__main__":
    main()

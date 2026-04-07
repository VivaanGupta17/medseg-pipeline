#!/usr/bin/env python3
"""Main training script for MedSeg Pipeline.

Supports:
    - Single-GPU and multi-GPU (torchrun / DistributedDataParallel)
    - Config-file-driven hyperparameter management
    - Checkpoint resuming
    - BraTS and LUNA16 datasets

Usage::

    # Single GPU
    python scripts/train.py \\
        --config configs/brain_tumor_config.yaml \\
        --data_root /data/brats2021 \\
        --output_dir experiments/run1

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 scripts/train.py \\
        --config configs/brain_tumor_config.yaml \\
        --data_root /data/brats2021

    # Resume from checkpoint
    python scripts/train.py \\
        --config configs/brain_tumor_config.yaml \\
        --resume experiments/run1/checkpoints/best_model.pth
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

# Ensure src is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import get_model
from data import BraTSDataset, LUNADataset, MedicalAugmentation
from training import Trainer, CombinedDiceCELoss, DiceLoss, TverskyLoss
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSeg Pipeline Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Override data_root from config.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["unet", "attention_unet"],
        help="Override model architecture from config.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable mixed precision training.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging.",
    )
    # Distributed training (set automatically by torchrun)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return a YAML config as a flat dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(local_rank: int) -> Tuple[int, int, torch.device]:
    """Initialise DDP if running with torchrun."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logger.info("DDP initialised: rank=%d world_size=%d", rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


from typing import Tuple


def build_loss(config: dict) -> torch.nn.Module:
    """Construct loss function from config."""
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "combined_dice_ce")
    num_classes = config.get("training", {}).get("num_classes", 4)

    class_weights = loss_cfg.get("class_weights")
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    if loss_type == "combined_dice_ce":
        return CombinedDiceCELoss(
            num_classes=num_classes,
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            ce_weight=loss_cfg.get("ce_weight", 0.5),
            ignore_index=loss_cfg.get("ignore_index", -100),
            class_weights=class_weights,
            dice_log=loss_cfg.get("dice_log", False),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )
    elif loss_type == "dice":
        return DiceLoss(num_classes=num_classes)
    elif loss_type == "tversky":
        return TverskyLoss(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_datasets(config: dict, data_root: str) -> Tuple[BraTSDataset, BraTSDataset]:
    """Construct training and validation datasets."""
    ds_cfg  = config.get("dataset", {})
    aug_cfg = config.get("augmentation", {})

    transform_train = (
        MedicalAugmentation(
            p=aug_cfg.get("p", 0.5),
            flip_axes=tuple(aug_cfg.get("flip_axes", [0, 1])),
            rotate_range=aug_cfg.get("rotate_range", 15.0),
            scale_range=tuple(aug_cfg.get("scale_range", [0.85, 1.15])),
            brightness_range=aug_cfg.get("brightness_range", 0.1),
            noise_std=aug_cfg.get("noise_std", 0.02),
            elastic_alpha=aug_cfg.get("elastic_alpha", 15.0),
            elastic_sigma=aug_cfg.get("elastic_sigma", 3.0),
        )
        if aug_cfg.get("enabled", True)
        else None
    )

    common_kwargs = dict(
        data_root=data_root,
        split_ratios=tuple(ds_cfg.get("split_ratios", [0.70, 0.15, 0.15])),
        seed=ds_cfg.get("seed", 42),
        slice_2d=ds_cfg.get("slice_2d", False),
        normalize=ds_cfg.get("normalize", True),
        cache_data=ds_cfg.get("cache_data", False),
    )

    train_ds = BraTSDataset(split="train", transform=transform_train, **common_kwargs)
    val_ds   = BraTSDataset(split="val",   transform=None,             **common_kwargs)
    return train_ds, val_ds


def build_dataloaders(
    train_ds,
    val_ds,
    config: dict,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Construct DataLoaders, adding DistributedSampler for DDP."""
    dl_cfg = config.get("dataloader", {})
    batch_size  = dl_cfg.get("batch_size", 2)
    num_workers = dl_cfg.get("num_workers", 8)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", True),
        persistent_workers=(num_workers > 0),
        prefetch_factor=dl_cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", True),
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.data_root:
        config.setdefault("dataset", {})["data_root"] = args.data_root
    if args.model:
        config.setdefault("model", {})["architecture"] = args.model
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("optimiser", {})["lr"] = args.lr
    if args.batch_size:
        config.setdefault("dataloader", {})["batch_size"] = args.batch_size
    if args.no_amp:
        config.setdefault("training", {})["use_amp"] = False

    # Seed and reproducibility
    repro_cfg = config.get("reproducibility", {})
    set_seed(repro_cfg.get("seed", 42), repro_cfg.get("deterministic", True))

    # Distributed setup
    rank, world_size, device = setup_distributed(args.local_rank)
    is_main = rank == 0

    if is_main:
        logger.info("Device: %s | World size: %d", device, world_size)
        logger.info("Config: %s", args.config)

    # Output directory
    out_cfg = config.get("output", {})
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(
            Path(out_cfg.get("base_dir", "experiments"))
            / out_cfg.get("experiment_name", "default_run")
        )
    if is_main:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

    # Build model
    model_cfg = config.get("model", {})
    model = get_model(
        model_cfg.get("architecture", "attention_unet"),
        in_channels=model_cfg.get("in_channels", 4),
        num_classes=model_cfg.get("num_classes", 4),
        spatial_dims=model_cfg.get("spatial_dims", 2),
        features=tuple(model_cfg.get("features", [64, 128, 256, 512])),
        norm_type=model_cfg.get("norm_type", "batch"),
        dropout_p=model_cfg.get("dropout_p", 0.1),
        residual=model_cfg.get("residual", False),
    )

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model: %s | Parameters: %.2fM",
            model_cfg.get("architecture"), n_params / 1e6,
        )

    # Wrap with DDP
    if world_size > 1:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        )

    # Datasets & loaders
    data_root = config.get("dataset", {}).get("data_root", "data/brats2021")
    train_ds, val_ds = build_datasets(config, data_root)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, config, rank, world_size)

    if is_main:
        logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # Loss function
    criterion = build_loss(config)

    # Flatten training config for Trainer
    train_cfg = config.get("training", {})
    optim_cfg = config.get("optimiser", {})
    sched_cfg = config.get("scheduler", {})

    trainer_config = {
        "lr":             optim_cfg.get("lr", 1e-3),
        "weight_decay":   optim_cfg.get("weight_decay", 1e-5),
        "epochs":         train_cfg.get("epochs", 200),
        "use_amp":        train_cfg.get("use_amp", True),
        "grad_clip":      train_cfg.get("grad_clip", 1.0),
        "patience":       train_cfg.get("patience", 20),
        "save_every":     train_cfg.get("save_every", 10),
        "log_interval":   train_cfg.get("log_interval", 20),
        "num_classes":    train_cfg.get("num_classes", 4),
        "scheduler":      sched_cfg.get("type", "cosine_warm"),
        "t0":             sched_cfg.get("t0", 50),
        "t_mult":         sched_cfg.get("t_mult", 2),
        "eta_min":        sched_cfg.get("eta_min", 1e-6),
        "plateau_patience": sched_cfg.get("plateau_patience", 10),
    }

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=trainer_config,
        output_dir=output_dir,
        device=device,
        rank=rank,
    )

    # Resume
    if args.resume:
        trainer.resume(args.resume)

    # Train
    results = trainer.train()

    if is_main:
        logger.info(
            "Training complete. Best validation mean Dice: %.4f",
            results["best_val_dice"],
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

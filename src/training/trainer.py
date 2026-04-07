"""Training loop for medical image segmentation models.

Features:
    - Mixed precision training (torch.cuda.amp) for 2× memory efficiency
    - Cosine annealing with warm restarts (SGDR)
    - Early stopping with configurable patience
    - Gradient clipping
    - Checkpoint saving (best model + periodic)
    - Distributed Data Parallel (DDP) ready
    - Comprehensive metric logging (TensorBoard + CSV)
    - Per-epoch validation with full metric suite

Usage::

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=CombinedDiceCELoss(num_classes=4),
        config=config,
        output_dir="experiments/run1",
    )
    trainer.train()
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..evaluation.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping callback monitoring a validation metric.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``'min'`` (lower is better) or ``'max'`` (higher is better).
        restore_best: If True, the monitor value is tracked and the best
            epoch is flagged for checkpoint restoration.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
        restore_best: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int = 0
        self.counter: int = 0
        self.stopped_epoch: int = 0

    def step(self, value: float, epoch: int) -> bool:
        """Update state and return True if training should stop.

        Args:
            value: Current validation metric value.
            epoch: Current epoch number.

        Returns:
            True if training should be stopped, False otherwise.
        """
        improved = (
            (self.mode == "max" and value > self.best_value + self.min_delta)
            or (self.mode == "min" and value < self.best_value - self.min_delta)
        )

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            logger.info(
                "Early stopping triggered at epoch %d. "
                "Best value: %.4f at epoch %d.",
                epoch, self.best_value, self.best_epoch,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training loop for segmentation models.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        criterion: Loss function (should return (total, dice, ce) or scalar).
        config: Dictionary of training hyperparameters (see below).
        output_dir: Directory for checkpoints, logs, and artefacts.
        device: Training device (auto-detected if not specified).
        rank: Rank for distributed training (0 = master process).

    Config keys:
        - ``lr``: Initial learning rate (default 1e-3).
        - ``weight_decay``: AdamW weight decay (default 1e-5).
        - ``epochs``: Maximum number of epochs (default 200).
        - ``grad_clip``: Max gradient norm (default 1.0, set 0 to disable).
        - ``use_amp``: Enable mixed precision (default True).
        - ``patience``: Early stopping patience in epochs (default 20).
        - ``save_every``: Save periodic checkpoints every N epochs (default 10).
        - ``scheduler``: LR scheduler type: ``'cosine_warm'`` or ``'plateau'``.
        - ``t0``: Cosine annealing T_0 period (default 50).
        - ``t_mult``: Cosine annealing T_mult multiplier (default 2).
        - ``eta_min``: Minimum LR for cosine schedule (default 1e-6).
        - ``num_classes``: Number of segmentation classes (default 4).
        - ``log_interval``: Log every N batches (default 10).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: Dict,
        output_dir: str = "experiments/default",
        device: Optional[torch.device] = None,
        rank: int = 0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.is_main = rank == 0

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        # Directories
        if self.is_main:
            (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Optimiser
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # LR Scheduler
        scheduler_type = config.get("scheduler", "cosine_warm")
        if scheduler_type == "cosine_warm":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.get("t0", 50),
                T_mult=config.get("t_mult", 2),
                eta_min=config.get("eta_min", 1e-6),
            )
        elif scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=config.get("plateau_patience", 10),
                factor=0.5,
                min_lr=config.get("eta_min", 1e-6),
            )
        else:
            self.scheduler = None
            logger.warning("No LR scheduler configured.")

        # Mixed precision
        self.use_amp = config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 20),
            mode="max",
        )

        # Metrics
        self.metrics_calc = SegmentationMetrics(
            num_classes=config.get("num_classes", 4)
        )

        # Logging
        if self.is_main:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs" / "tensorboard"))
            self._csv_path = self.output_dir / "logs" / "training_history.csv"
            self._init_csv()

        # State
        self.start_epoch = 0
        self.best_val_dice = 0.0
        self.history: List[Dict] = []

        logger.info(
            "Trainer initialised: device=%s  amp=%s  epochs=%d  lr=%.2e",
            self.device, self.use_amp, config.get("epochs", 200), config.get("lr", 1e-3),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """Run the full training loop.

        Returns:
            Dict with training history and best validation metrics.
        """
        n_epochs = self.config.get("epochs", 200)
        logger.info("Starting training for %d epochs.", n_epochs)

        for epoch in range(self.start_epoch, n_epochs):
            epoch_start = time.perf_counter()

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._val_epoch(epoch)

            epoch_time = time.perf_counter() - epoch_start

            # LR step
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["mean_dice"])
            elif self.scheduler is not None:
                self.scheduler.step(epoch + 1)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            if self.is_main:
                self._log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time)
                self._write_csv(epoch, train_metrics, val_metrics, current_lr)

                # Checkpoint: best model
                if val_metrics["mean_dice"] > self.best_val_dice:
                    self.best_val_dice = val_metrics["mean_dice"]
                    self._save_checkpoint(epoch, val_metrics, tag="best")
                    logger.info(
                        "New best model — epoch %d  mean_dice=%.4f",
                        epoch, self.best_val_dice,
                    )

                # Periodic checkpoint
                if (epoch + 1) % self.config.get("save_every", 10) == 0:
                    self._save_checkpoint(epoch, val_metrics, tag=f"epoch_{epoch:04d}")

            # Early stopping
            if self.early_stopping.step(val_metrics["mean_dice"], epoch):
                logger.info("Early stopping. Training complete.")
                break

        if self.is_main:
            self.writer.close()
            self._save_final_summary()

        return {
            "best_val_dice": self.best_val_dice,
            "history": self.history,
        }

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to ``.pth`` checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_dice = checkpoint.get("best_val_dice", 0.0)
        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(
            "Resumed training from %s at epoch %d.", checkpoint_path, self.start_epoch
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_dice_loss = 0.0
        total_ce_loss = 0.0
        n_batches = len(self.train_loader)
        log_interval = self.config.get("log_interval", 10)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            masks  = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss_output = self.criterion(logits, masks)
                if isinstance(loss_output, tuple):
                    loss, dice_loss, ce_loss = loss_output
                else:
                    loss = loss_output
                    dice_loss = ce_loss = loss_output

            self.scaler.scale(loss).backward()

            # Gradient clipping
            grad_clip = self.config.get("grad_clip", 1.0)
            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss      += loss.item()
            total_dice_loss += dice_loss.item()
            total_ce_loss   += ce_loss.item()

            if self.is_main and (batch_idx + 1) % log_interval == 0:
                logger.debug(
                    "  Epoch %d [%d/%d]  loss=%.4f  dice_loss=%.4f  ce_loss=%.4f",
                    epoch, batch_idx + 1, n_batches,
                    loss.item(), dice_loss.item(), ce_loss.item(),
                )

        return {
            "loss":      total_loss / n_batches,
            "dice_loss": total_dice_loss / n_batches,
            "ce_loss":   total_ce_loss / n_batches,
        }

    def _val_epoch(self, epoch: int) -> Dict:
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                masks  = batch["mask"].to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss_output = self.criterion(logits, masks)
                    loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output

                total_loss += loss.item()

                preds = logits.argmax(dim=1).cpu().numpy()
                tgts  = masks.cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(tgts)

        # Compute full metric suite
        import numpy as np
        all_preds_np   = np.stack(all_preds, axis=0)
        all_targets_np = np.stack(all_targets, axis=0)
        metrics = self.metrics_calc.compute_batch(all_preds_np, all_targets_np)

        metrics["val_loss"] = total_loss / max(len(self.val_loader), 1)
        return metrics

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        epoch_time: float,
    ) -> None:
        """Write metrics to TensorBoard and console."""
        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | "
            "val_mean_dice=%.4f | lr=%.2e | %.1fs",
            epoch,
            train_metrics["loss"],
            val_metrics.get("val_loss", float("nan")),
            val_metrics.get("mean_dice", float("nan")),
            lr,
            epoch_time,
        )

        self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/val", val_metrics.get("val_loss", 0), epoch)
        self.writer.add_scalar("Dice/mean_dice", val_metrics.get("mean_dice", 0), epoch)
        self.writer.add_scalar("LR", lr, epoch)

        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"Val/{k}", v, epoch)

        record = {"epoch": epoch, "lr": lr, **train_metrics, **val_metrics, "time": epoch_time}
        self.history.append(record)

    def _save_checkpoint(self, epoch: int, val_metrics: Dict, tag: str) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_metrics": val_metrics,
            "best_val_dice": self.best_val_dice,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        ckpt_path = self.output_dir / "checkpoints" / f"{tag}_model.pth"
        torch.save(checkpoint, str(ckpt_path))
        logger.debug("Saved checkpoint: %s", ckpt_path)

    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "train_loss", "val_loss", "mean_dice",
                             "wt_dice", "tc_dice", "et_dice", "hd95", "time"])

    def _write_csv(
        self, epoch: int, train: Dict, val: Dict, lr: float
    ) -> None:
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{lr:.2e}",
                f"{train.get('loss', 0):.4f}",
                f"{val.get('val_loss', 0):.4f}",
                f"{val.get('mean_dice', 0):.4f}",
                f"{val.get('class_dice', {}).get(1, 0):.4f}",  # WT ~ NCR
                f"{val.get('class_dice', {}).get(2, 0):.4f}",  # TC ~ Edema
                f"{val.get('class_dice', {}).get(3, 0):.4f}",  # ET
                f"{val.get('hausdorff_95', 0):.2f}",
                f"{time.time():.0f}",
            ])

    def _save_final_summary(self) -> None:
        summary = {
            "best_val_dice": self.best_val_dice,
            "early_stop_epoch": self.early_stopping.stopped_epoch,
            "best_epoch": self.early_stopping.best_epoch,
            "config": self.config,
        }
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Training summary saved to %s", self.output_dir)

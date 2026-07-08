"""Main training loop with mixed precision, feature caching, and multi-task optimization."""

import gc
import math
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from ..config import Config
from ..data.datasets import MoleculeDataset, collate_fn, DATASET_CONFIGS
from ..features.pipeline import FeaturePipeline
from ..models.cot_moe_model import CoTMoEModel
from ..llm.cot_generator import CoTPlaceholder
from .losses import UncertaintyWeightedLoss
from .metrics import compute_metrics, aggregate_metrics
from .optimizer import build_optimizer_and_scheduler
from ..utils.logging import setup_logger, AverageMeter
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """Full training loop for CoT-MoE."""

    def __init__(
        self,
        model: CoTMoEModel,
        feature_pipeline: FeaturePipeline,
        config: Config,
        train_dataset: MoleculeDataset,
        val_dataset: Optional[MoleculeDataset] = None,
    ):
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.config = config

        cfg = config.training
        self.batch_size = cfg.batch_size
        self.grad_accumulation = cfg.grad_accumulation
        self.max_epochs = cfg.max_epochs
        self.freeze_encoder_epochs = getattr(cfg, 'freeze_encoder_epochs', 15)
        self.early_stop_patience = getattr(cfg, 'early_stop_patience', 25)
        self.grad_clip = cfg.grad_clip
        self.precision = cfg.precision
        self.log_interval = cfg.log_interval
        self.eval_interval = cfg.eval_interval
        self.save_interval = cfg.save_interval
        self.output_dir = Path(config.output_dir)

        self.device = torch.device(config.device if config.device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Move model to device
        self.model = self.model.to(self.device)
        self.feature_pipeline.init_egnn(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=collate_fn,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                collate_fn=collate_fn,
            )

        # Total steps estimate
        self.total_steps = len(self.train_loader) * self.max_epochs // self.grad_accumulation

        # Optimizer & scheduler
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            self.model,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
            total_steps=self.total_steps,
        )

        # Uncertainty weighting (task-type-aware: regression vs classification)
        all_tasks = list(DATASET_CONFIGS.keys())
        regression_tasks = [t for t in all_tasks
                           if DATASET_CONFIGS[t].get("task") == "regression"]
        self.uncertainty_loss = UncertaintyWeightedLoss(all_tasks, regression_tasks)

        # Mixed precision
        self.use_amp = self.precision in ("bf16", "fp16") and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=(self.precision == "fp16"))

        # CoT placeholder (LLM is optional)
        self.model.cot_generator = CoTPlaceholder(cot_output_dim=config.model.cot_output_dim)

        # Logger
        self.logger = setup_logger(log_file=str(self.output_dir / "train.log"))

        # State tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.features_cached = False

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_features(self, batch: Dict) -> tuple:
        """Compute or load cached features for a batch."""
        smiles_list = batch["smiles"]
        mol_indices = batch["mol_indices"].tolist()

        features_list = []
        for smi, midx in zip(smiles_list, mol_indices):
            feats = self.feature_pipeline.compute_single(smi, mol_idx=midx)
            if feats is None:
                self.logger.warning(f"Failed to process SMILES: {smi}")
                continue
            features_list.append(feats)

        if not features_list:
            raise RuntimeError("All molecules in batch failed feature extraction")

        return self.feature_pipeline.collate_features(features_list, self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        meters = {
            "loss": AverageMeter(),
            "align_loss": AverageMeter(),
            "aux_loss": AverageMeter(),
            "dispersion_loss": AverageMeter(),
        }
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                features = self._get_features(batch)
            except RuntimeError:
                continue

            # Forward pass
            with autocast(device_type=self.device.type, enabled=self.use_amp,
                          dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16):
                predictions = self.model(features, smiles=batch["smiles"])
                task_losses = self.model.get_task_losses(predictions, batch["labels"])
                weighted_losses = self.uncertainty_loss(task_losses)
                loss = weighted_losses["_total"] / self.grad_accumulation

            # Backward
            if self.use_amp and self.precision == "fp16":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step
            if (batch_idx + 1) % self.grad_accumulation == 0:
                if self.use_amp and self.precision == "fp16":
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self.use_amp and self.precision == "fp16":
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Logging
            meters["loss"].update(task_losses.get("_total", 0.0).item())
            meters["align_loss"].update(task_losses.get("_align", 0.0).item() if isinstance(task_losses.get("_align"), torch.Tensor) else 0.0)
            meters["aux_loss"].update(task_losses.get("_aux", 0.0).item() if isinstance(task_losses.get("_aux"), torch.Tensor) else 0.0)
            meters["dispersion_loss"].update(task_losses.get("_mph_dispersion", 0.0).item() if isinstance(task_losses.get("_mph_dispersion"), torch.Tensor) else 0.0)

            if batch_idx % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Epoch {self.current_epoch} | Step {self.global_step} | "
                    f"Loss {meters['loss'].avg:.4f} | Align {meters['align_loss'].avg:.4f} | "
                    f"Aux {meters['aux_loss'].avg:.4f} | Disp {meters['dispersion_loss'].avg:.4f} | LR {lr:.2e}"
                )

            # Validation
            if self.val_loader is not None and self.global_step % self.eval_interval == 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation: {val_metrics}")
                self.model.train()

            # Checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop."""
        self.model.eval()
        all_preds: Dict[str, list] = {}
        all_labels: Dict[str, list] = {}
        task_types: Dict[str, str] = {}

        for batch in self.val_loader:
            try:
                features = self._get_features(batch)
            except RuntimeError:
                continue

            predictions = self.model(features, smiles=batch["smiles"])

            for ds_name, pred in predictions.items():
                if ds_name.startswith("_"):
                    continue
                p = pred.cpu().numpy()
                if ds_name not in all_preds:
                    all_preds[ds_name] = []
                    all_labels[ds_name] = []
                    task_types[ds_name] = DATASET_CONFIGS[ds_name]["task"]
                all_preds[ds_name].append(p)
                if ds_name in batch["labels"]:
                    all_labels[ds_name].append(batch["labels"][ds_name].cpu().numpy())

        # Concatenate
        preds_dict = {}
        labels_dict = {}
        for ds_name in all_preds:
            preds_dict[ds_name] = np.concatenate(all_preds[ds_name], axis=0)
            if all_labels[ds_name]:
                labels_dict[ds_name] = np.concatenate(all_labels[ds_name], axis=0)

        metrics = compute_metrics(preds_dict, labels_dict, task_types)
        agg = aggregate_metrics(metrics, task_types)
        metrics.update(agg)

        # Track best by avg_rmse (primary metric for FreeSolv/ESOL/Lipo)
        # QM7 MAE is tracked separately but not used for model selection.
        if "avg_rmse" in metrics:
            current = metrics["avg_rmse"]
            if current < self.best_metric:
                self.best_metric = current
                self.logger.info(f"New best RMSE: {current:.4f}")

        return metrics

    def train(self) -> None:
        """Run full training loop with two-stage freeze + early stopping."""
        self.logger.info(f"Starting training: {self.max_epochs} epochs, device={self.device}")
        self.logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Freeze encoders: first {self.freeze_encoder_epochs} epochs")

        # ── Two-stage training ──────────────────────────────────────
        encoder_prefixes = ("mph_encoder", "pls_encoder", "egnn_encoder")
        if self.freeze_encoder_epochs > 0:
            self.logger.info(f"Stage 1: freezing encoders for epochs 0-{self.freeze_encoder_epochs - 1}")
            for name, p in self.model.named_parameters():
                if any(pre in name for pre in encoder_prefixes):
                    p.requires_grad = False

        no_improve_count = 0

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Unfreeze at boundary
            if self.freeze_encoder_epochs > 0 and epoch == self.freeze_encoder_epochs:
                self.logger.info(f"Stage 2: unfreezing encoders at epoch {epoch}")
                for name, p in self.model.named_parameters():
                    if any(pre in name for pre in encoder_prefixes):
                        p.requires_grad = True

            start_time = time.time()
            metrics = self.train_epoch()
            elapsed = time.time() - start_time

            self.logger.info(
                f"Epoch {epoch} done | Time {elapsed:.1f}s | "
                f"Loss {metrics['loss']:.4f} | Align {metrics['align_loss']:.4f}"
            )

            # ── Validation + early stopping ─────────────────────────
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.logger.info(f"Epoch {epoch} validation: {val_metrics}")

                current = val_metrics.get("avg_rmse", float("inf"))
                if current < self.best_metric - 1e-6:
                    self.best_metric = current
                    no_improve_count = 0
                    self.logger.info(f"New best RMSE: {current:.4f}")
                    self.save_checkpoint(filename="best.pt")
                else:
                    no_improve_count += 1
                    self.logger.info(f"No improvement for {no_improve_count}/{self.early_stop_patience} epochs")

                if no_improve_count >= self.early_stop_patience:
                    self.logger.info(f"Early stop at epoch {epoch} (no improvement for {self.early_stop_patience} epochs)")
                    break

        self.save_checkpoint(filename="final.pt")
        self.logger.info(f"Training complete. Best RMSE: {self.best_metric:.4f}")

    def save_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        path = str(self.output_dir / filename)
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            metrics={"best_metric": self.best_metric},
            path=path,
            extra={"standardizer": self.train_loader.dataset.get_standardizer().to_dict()},
        )
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        info = load_checkpoint(path, self.model, self.optimizer, self.scheduler, str(self.device))
        self.current_epoch = info["epoch"]
        self.global_step = info["global_step"]
        self.best_metric = info["metrics"].get("best_metric", float("inf"))
        self.logger.info(f"Checkpoint loaded: epoch={info['epoch']}, step={info['global_step']}")

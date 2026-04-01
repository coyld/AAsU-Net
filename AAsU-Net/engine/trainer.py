from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..engine.callbacks import CSVLogger, EarlyStopping
from ..losses.deep_supervision import DeepSupervisionLoss
from ..utils.checkpoint import resume_from_checkpoint, save_checkpoint
from ..utils.logging import setup_logger
from ..utils.misc import AverageMeter, ensure_dir, human_readable_seconds
from ..utils.seed import seed_everything
from .evaluator import evaluate_loader


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: DeepSupervisionLoss,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.output_dir = ensure_dir(cfg.project.output_dir)
        self.ckpt_dir = ensure_dir(self.output_dir / "checkpoints")
        self.logger = setup_logger("aasunet", self.output_dir / "train.log")
        self.csv_logger = CSVLogger(self.output_dir / "metrics.csv")
        self.scaler = torch.amp.GradScaler(enabled=(cfg.train.amp and device.type == "cuda"))
        self.early_stopping = EarlyStopping(cfg.train.early_stopping_patience, mode="max")

        if cfg.train.resume:
            ckpt = resume_from_checkpoint(
                cfg.train.resume,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                map_location=device,
            )
            self.start_epoch = int(ckpt.get("epoch", 0)) + 1
            self.best_metric = float(ckpt.get("best_metric", float("-inf")))
            self.logger.info("Resumed from %s (epoch=%d, best_metric=%.4f)", cfg.train.resume, self.start_epoch, self.best_metric)
        else:
            self.start_epoch = 1
            self.best_metric = float("-inf")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        total_start = time.time()
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            summary = {"epoch": epoch, "train/loss": train_loss}

            if val_loader is not None:
                val_metrics = evaluate_loader(self.model, val_loader, self.cfg, self.device)
                summary.update({f"val/{k}": v for k, v in val_metrics.items()})
                monitor = val_metrics.get("tumor/dice") or val_metrics.get("kidney/dice") or max(val_metrics.values())
                if monitor > self.best_metric:
                    self.best_metric = monitor
                    save_checkpoint(
                        self.ckpt_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        best_metric=self.best_metric,
                        config=self.cfg.to_dict(),
                    )
                    self.logger.info("Saved new best checkpoint at epoch %d with metric %.4f", epoch, self.best_metric)

                if self.early_stopping.step(monitor):
                    self.logger.info("Early stopping triggered at epoch %d.", epoch)
                    self.csv_logger.log(summary)
                    break

            if self.cfg.train.save_last:
                save_checkpoint(
                    self.ckpt_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    config=self.cfg.to_dict(),
                )

            if epoch % self.cfg.train.checkpoint_every == 0:
                save_checkpoint(
                    self.ckpt_dir / f"epoch_{epoch:04d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    config=self.cfg.to_dict(),
                )

            self.csv_logger.log(summary)
            self.logger.info("Epoch %d summary: %s", epoch, summary)

        elapsed = time.time() - total_start
        self.logger.info("Training finished in %s", human_readable_seconds(elapsed))

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        meter = AverageMeter()
        start = time.time()

        iterator = iter(train_loader)
        for step in range(1, self.cfg.train.iterations_per_epoch + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=(self.cfg.train.amp and self.device.type == "cuda")):
                outputs = self.model(image)
                loss = self.criterion(outputs["deep_supervision"], label)

            self.scaler.scale(loss).backward()

            if self.cfg.train.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            meter.update(float(loss.item()))
            if step % self.cfg.train.log_interval == 0 or step == self.cfg.train.iterations_per_epoch:
                elapsed = time.time() - start
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    "epoch=%d step=%d/%d loss=%.4f lr=%.6f elapsed=%s",
                    epoch,
                    step,
                    self.cfg.train.iterations_per_epoch,
                    meter.avg,
                    lr,
                    human_readable_seconds(elapsed),
                )

        return meter.avg

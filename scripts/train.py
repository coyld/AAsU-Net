#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aasunet.config import load_config
from aasunet.data.dataset import CTVolumeDataset, load_manifest
from aasunet.engine.trainer import Trainer
from aasunet.losses.deep_supervision import DeepSupervisionLoss
from aasunet.losses.hybrid import DiceCrossEntropyLoss
from aasunet.models.factory import build_model
from aasunet.optim.schedulers import PolyLRScheduler
from aasunet.utils.seed import seed_everything


def build_optimizer(cfg, model):
    if cfg.optimizer.name.lower() != "sgd":
        raise ValueError("This repo follows the paper and defaults to SGD.")
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        nesterov=cfg.optimizer.nesterov,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AAsU-Net.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    if args.train_manifest:
        cfg.data.train_manifest = str(args.train_manifest)
    if args.val_manifest:
        cfg.data.val_manifest = str(args.val_manifest)

    if not cfg.data.train_manifest:
        raise ValueError("Training manifest is required.")
    if not cfg.data.val_manifest:
        raise ValueError("Validation manifest is required.")

    seed_everything(cfg.project.seed, deterministic=cfg.runtime.deterministic)
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")

    train_dataset = CTVolumeDataset(load_manifest(cfg.data.train_manifest), cfg, training=True)
    val_dataset = CTVolumeDataset(load_manifest(cfg.data.val_manifest), cfg, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        num_workers=max(cfg.data.num_workers // 2, 0),
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    max_steps = cfg.train.epochs * cfg.train.iterations_per_epoch
    scheduler = PolyLRScheduler(optimizer, max_steps=max_steps, power=cfg.optimizer.poly_power)
    base_loss = DiceCrossEntropyLoss(
        dice_weight=cfg.loss.dice_weight,
        ce_weight=cfg.loss.ce_weight,
        include_background=cfg.loss.include_background,
    )
    criterion = DeepSupervisionLoss(base_loss, weights=cfg.loss.deep_supervision_weights)

    trainer = Trainer(cfg, model, optimizer, scheduler, criterion, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

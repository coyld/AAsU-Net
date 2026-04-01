from __future__ import annotations

from typing import Any, Dict

import torch

from ..config import ExperimentConfig
from .aasunet import AAsUNet


def build_model(cfg: ExperimentConfig) -> AAsUNet:
    return AAsUNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        encoder_channels=cfg.model.encoder_channels,
        conv_mode=cfg.model.conv_mode,
        use_csff=cfg.model.use_csff,
        reduction=cfg.model.reduction,
        leakiness=cfg.model.leakiness,
        deep_supervision=cfg.model.deep_supervision,
        dropout=cfg.model.dropout,
    )


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(p.numel() for p in params)

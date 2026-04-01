from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice import SoftDiceLoss


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = SoftDiceLoss(include_background=include_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_term = self.dice(logits, target)
        ce_term = F.cross_entropy(logits, target.long())
        return self.dice_weight * dice_term + self.ce_weight * ce_term

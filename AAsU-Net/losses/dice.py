from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_labels(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


class SoftDiceLoss(nn.Module):
    def __init__(self, include_background: bool = True, smooth: float = 1e-5) -> None:
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        target_1h = one_hot_labels(target, logits.shape[1]).type_as(probs)

        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]

        dims = (0, 2, 3, 4)
        intersection = (probs * target_1h).sum(dims)
        denominator = probs.sum(dims) + target_1h.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()

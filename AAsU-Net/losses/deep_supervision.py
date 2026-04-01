from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, weights: Sequence[float] | None = None) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = list(weights) if weights is not None else None

    def forward(self, outputs: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if len(outputs) == 0:
            raise ValueError("DeepSupervisionLoss expected at least one output tensor.")

        if self.weights is None:
            weights = [1.0 / (2**i) for i in range(len(outputs))]
        else:
            weights = list(self.weights[: len(outputs)])
            if len(weights) < len(outputs):
                weights.extend([weights[-1]] * (len(outputs) - len(weights)))

        total_weight = float(sum(weights))
        total = outputs[0].new_tensor(0.0)

        for weight, logits in zip(weights, outputs):
            resized_target = target
            if logits.shape[2:] != target.shape[1:]:
                resized_target = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=logits.shape[2:],
                    mode="nearest",
                ).squeeze(1).long()
            total = total + float(weight) * self.base_loss(logits, resized_target)

        return total / max(total_weight, 1e-8)

from __future__ import annotations

from typing import Any, Dict

import torch


class PolyLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, max_steps: int, power: float = 0.9) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.optimizer = optimizer
        self.max_steps = int(max_steps)
        self.power = float(power)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_idx = 0

    def _compute_lr(self, step: int, base_lr: float) -> float:
        coeff = max(1.0 - (step / float(self.max_steps)), 0.0) ** self.power
        return base_lr * coeff

    def step(self) -> None:
        self.step_idx += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._compute_lr(self.step_idx, base_lr)

    def state_dict(self) -> Dict[str, Any]:
        return {"step_idx": self.step_idx, "max_steps": self.max_steps, "power": self.power, "base_lrs": self.base_lrs}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.step_idx = int(state_dict["step_idx"])
        self.max_steps = int(state_dict["max_steps"])
        self.power = float(state_dict["power"])
        self.base_lrs = list(state_dict["base_lrs"])
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self._compute_lr(self.step_idx, base_lr)

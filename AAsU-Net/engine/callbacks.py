from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: Mapping[str, object]) -> None:
        row_dict = dict(row)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row_dict)

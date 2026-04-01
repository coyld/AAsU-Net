from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..utils.misc import write_jsonl


def train_val_split(
    records: Sequence[Mapping],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rows = [dict(r) for r in records]
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_ratio)))
    val = rows[:n_val]
    train = rows[n_val:]
    return train, val


def save_split(train: Sequence[Mapping], val: Sequence[Mapping], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")

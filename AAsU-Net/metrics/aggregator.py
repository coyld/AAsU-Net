from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Mapping


class MetricAggregator:
    def __init__(self) -> None:
        self.values: DefaultDict[str, list[float]] = defaultdict(list)

    def update(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self.values[key].append(float(value))

    def summary(self) -> Dict[str, float]:
        return {key: sum(values) / max(len(values), 1) for key, values in self.values.items()}

    def reset(self) -> None:
        self.values.clear()

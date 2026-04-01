from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_slice_triplet(
    image: np.ndarray,
    label: np.ndarray | None,
    prediction: np.ndarray | None,
    output_path: str | Path,
    axis: int = 0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 4:
        image = image[0]

    idx = image.shape[axis] // 2
    take = [slice(None)] * 3
    take[axis] = idx
    sl = tuple(take)

    fig = plt.figure(figsize=(12, 4))
    panes = [image[sl], label[sl] if label is not None else None, prediction[sl] if prediction is not None else None]
    titles = ["image", "label", "prediction"]

    for i, (pane, title) in enumerate(zip(panes, titles), start=1):
        ax = fig.add_subplot(1, 3, i)
        if pane is None:
            ax.axis("off")
            ax.set_title(title)
            continue
        ax.imshow(pane, cmap="gray" if i == 1 else None)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

import numpy as np

from aasunet.metrics.overlap import binary_dice, binary_iou
from aasunet.metrics.surface import binary_asd, binary_hd95


def test_metrics_identity():
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    assert abs(binary_dice(mask, mask) - 1.0) < 1e-6
    assert abs(binary_iou(mask, mask) - 1.0) < 1e-6
    assert abs(binary_asd(mask, mask, (1.0, 1.0, 1.0))) < 1e-6
    assert abs(binary_hd95(mask, mask, (1.0, 1.0, 1.0))) < 1e-6

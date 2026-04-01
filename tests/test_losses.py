import torch

from aasunet.losses.deep_supervision import DeepSupervisionLoss
from aasunet.losses.hybrid import DiceCrossEntropyLoss


def test_hybrid_loss_positive():
    loss_fn = DiceCrossEntropyLoss()
    ds = DeepSupervisionLoss(loss_fn, weights=[1, 0.5, 0.25, 0.125, 0.0625])
    logits = [
        torch.randn(2, 3, 16, 64, 64),
        torch.randn(2, 3, 8, 32, 32),
        torch.randn(2, 3, 4, 16, 16),
        torch.randn(2, 3, 2, 8, 8),
        torch.randn(2, 3, 1, 4, 4),
    ]
    target = torch.randint(0, 3, (2, 16, 64, 64))
    loss = ds(logits, target)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0

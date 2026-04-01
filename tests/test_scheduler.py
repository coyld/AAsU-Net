import torch

from aasunet.optim.schedulers import PolyLRScheduler


def test_poly_scheduler_monotonic():
    model = torch.nn.Conv3d(1, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = PolyLRScheduler(optimizer, max_steps=10, power=0.9)
    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))

import torch

from aasunet.config import ExperimentConfig
from aasunet.models.factory import build_model


def test_model_forward_shapes_across_modes():
    for mode, use_csff in [("aas", True), ("aas", False), ("sum", False), ("separable", False), ("standard", False)]:
        cfg = ExperimentConfig()
        cfg.data.patch_size = [16, 64, 64]
        cfg.model.encoder_channels = [4, 8, 16, 24, 32, 32]
        cfg.model.conv_mode = mode
        cfg.model.use_csff = use_csff
        model = build_model(cfg)
        x = torch.randn(1, 1, 16, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out["logits"].shape == (1, cfg.data.num_classes, 16, 64, 64)
        assert len(out["deep_supervision"]) == 5

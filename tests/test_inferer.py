import torch

from aasunet.engine.inferer import sliding_window_inference


class DummyPredictor(torch.nn.Module):
    def forward(self, x):
        return {"logits": torch.cat([x, x, x], dim=1)}


def test_sliding_window_output_shape():
    predictor = DummyPredictor()
    x = torch.randn(1, 1, 16, 32, 32)
    out = sliding_window_inference(x, roi_size=(8, 16, 16), sw_batch_size=2, predictor=predictor, overlap=0.5)
    assert out.shape == (1, 3, 16, 32, 32)

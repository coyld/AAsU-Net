from pathlib import Path

from aasunet.config import load_config


def test_config_override(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("project:\n  output_dir: outputs/test\ntrain:\n  epochs: 10\n", encoding="utf-8")
    cfg = load_config(path, overrides=["train.epochs=12", "optimizer.lr=0.005"])
    assert cfg.train.epochs == 12
    assert abs(cfg.optimizer.lr - 0.005) < 1e-9

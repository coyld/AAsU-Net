from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml


def _as_path(path: str | Path | None) -> str | None:
    return None if path is None else str(path)


@dataclass
class ProjectConfig:
    name: str = "AAsU-Net"
    output_dir: str = "outputs/kits19_aasunet"
    seed: int = 42


@dataclass
class DataConfig:
    train_manifest: str | None = None
    val_manifest: str | None = None
    test_manifest: str | None = None
    input_channels: int = 1
    num_classes: int = 3
    patch_size: List[int] = field(default_factory=lambda: [64, 128, 128])
    target_spacing: List[float] = field(default_factory=lambda: [3.22, 1.62, 1.62])
    intensity_clip: List[float] = field(default_factory=lambda: [-75.0, 293.0])
    zscore: bool = True
    positive_sample_ratio: float = 0.75
    foreground_labels: List[int] = field(default_factory=lambda: [1, 2])
    cache_in_memory: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    label_map: Dict[int, int] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    enabled: bool = True
    flip_prob: float = 0.5
    rotate90_prob: float = 0.5
    small_rotate_prob: float = 0.2
    rotate_limit_deg: float = 15.0
    zoom_prob: float = 0.2
    zoom_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    brightness_prob: float = 0.15
    brightness_delta: float = 0.1
    contrast_prob: float = 0.15
    contrast_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    gamma_prob: float = 0.15
    gamma_range: List[float] = field(default_factory=lambda: [0.7, 1.5])


@dataclass
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 3
    encoder_channels: List[int] = field(default_factory=lambda: [24, 48, 96, 192, 320, 320])
    conv_mode: str = "aas"
    use_csff: bool = True
    reduction: int = 4
    leakiness: float = 0.01
    deep_supervision: bool = True
    dropout: float = 0.0


@dataclass
class LossConfig:
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    include_background: bool = True
    deep_supervision_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125, 0.0625]
    )


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 1e-2
    momentum: float = 0.99
    weight_decay: float = 3e-5
    nesterov: bool = True
    poly_power: float = 0.9


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 2
    iterations_per_epoch: int = 250
    amp: bool = True
    grad_clip: float | None = 12.0
    early_stopping_patience: int = 20
    checkpoint_every: int = 20
    resume: str | None = None
    output_dir: str | None = None
    save_last: bool = True
    log_interval: int = 20


@dataclass
class ValidationConfig:
    batch_size: int = 1
    sliding_window_batch: int = 1
    overlap: float = 0.5
    use_gaussian: bool = True
    save_predictions: bool = False
    regions: List[str] = field(default_factory=lambda: ["kidney", "tumor"])


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    benchmark: bool = False
    deterministic: bool = False
    num_threads: int | None = None


@dataclass
class ExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _set_by_dotted_key(payload: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = payload
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def apply_overrides(config: ExperimentConfig, overrides: Sequence[str] | None) -> ExperimentConfig:
    if not overrides:
        return config
    payload = config.to_dict()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected dotted.path=value")
        key, raw_value = item.split("=", 1)
        _set_by_dotted_key(payload, key, _parse_override_value(raw_value))
    return from_dict(payload)


def from_dict(payload: Mapping[str, Any]) -> ExperimentConfig:
    def _construct(cls, data: Mapping[str, Any]):
        kwargs = {}
        for field_def in dataclasses.fields(cls):
            field_value = data.get(field_def.name)
            if field_value is None:
                continue
            if dataclasses.is_dataclass(field_def.type):
                kwargs[field_def.name] = _construct(field_def.type, field_value)
            else:
                kwargs[field_def.name] = field_value
        return cls(**kwargs)

    # explicit construction keeps mypy / IDE friendly defaults
    return ExperimentConfig(
        project=ProjectConfig(**payload.get("project", {})),
        data=DataConfig(**payload.get("data", {})),
        augmentation=AugmentationConfig(**payload.get("augmentation", {})),
        model=ModelConfig(**payload.get("model", {})),
        loss=LossConfig(**payload.get("loss", {})),
        optimizer=OptimizerConfig(**payload.get("optimizer", {})),
        train=TrainConfig(**payload.get("train", {})),
        validation=ValidationConfig(**payload.get("validation", {})),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
    )


def load_config(path: str | Path | None, overrides: Sequence[str] | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        merged = _deep_update(config.to_dict(), payload)
        config = from_dict(merged)
    config.model.in_channels = config.data.input_channels
    config.model.out_channels = config.data.num_classes
    if config.train.output_dir:
        config.project.output_dir = config.train.output_dir
    config = apply_overrides(config, overrides)
    config.model.in_channels = config.data.input_channels
    config.model.out_channels = config.data.num_classes
    if config.train.output_dir:
        config.project.output_dir = config.train.output_dir
    return config

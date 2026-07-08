"""Configuration system using dataclasses, loaded from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    align_dim: int = 1024
    num_experts: int = 8
    top_k: int = 2
    expert_capacity_factor: float = 1.2
    llm_name: str = "meta-llama/Meta-Llama-3-8B"
    llm_load_mode: str = "8bit"  # "8bit" | "bf16"
    lora_r: int = 32
    lora_alpha: int = 64
    num_virtual_tokens: int = 8
    cot_output_dim: int = 1024
    cot_max_new_tokens: int = 128
    mph_resolution: int = 80
    mph_num_scales: int = 3
    mph_sigma_values: list = field(default_factory=lambda: [0.25, 0.5, 1.0])
    mph_dispersion_weight: float = 0.01
    pls_top_k: int = 256
    pls_num_filtrations: int = 100
    egnn_layers: int = 4
    egnn_hidden: int = 128
    egnn_frozen_coords: bool = True
    rsas_embed_dim: int = 64
    contrastive_temp: float = 0.07
    use_flash_attention: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 16
    grad_accumulation: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 80              # was 100; 80 sufficient for MoleculeNet
    freeze_encoder_epochs: int = 15   # stage-1 duration before full fine-tune
    early_stop_patience: int = 25     # stop if no val improvement for N epochs
    grad_clip: float = 1.0
    precision: str = "bf16"  # "bf16" | "fp16" | "fp32"
    pin_memory: bool = True
    num_workers: int = 4
    seed: int = 42
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 2000


@dataclass
class DataConfig:
    data_dir: str = "./data"
    cache_dir: str = "./cache/features"
    datasets: list = field(default_factory=lambda: [
        "bbbp", "bace", "clintox", "sider", "tox21",
        "freesolv", "esol", "lipo", "qm7",
    ])
    regression_datasets: list = field(default_factory=lambda: [
        "freesolv", "esol", "lipo", "qm7",
    ])
    classification_datasets: list = field(default_factory=lambda: [
        "bbbp", "bace", "clintox", "sider", "tox21",
    ])
    num_classes: dict = field(default_factory=lambda: {
        "bbbp": 1, "bace": 1, "clintox": 2,
        "sider": 27, "tox21": 12,
    })


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "./outputs"
    device: str = "auto"


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()
    if "model" in raw:
        cfg.model = ModelConfig(**{k: v for k, v in raw["model"].items() if k in ModelConfig.__dataclass_fields__})
    if "training" in raw:
        cfg.training = TrainingConfig(**{k: v for k, v in raw["training"].items() if k in TrainingConfig.__dataclass_fields__})
    if "data" in raw:
        cfg.data = DataConfig(**{k: v for k, v in raw["data"].items() if k in DataConfig.__dataclass_fields__})
    if "output_dir" in raw:
        cfg.output_dir = raw["output_dir"]
    if "device" in raw:
        cfg.device = raw["device"]
    return cfg


def save_config(cfg: Config, path: str) -> None:
    import dataclasses
    raw = {
        "model": dataclasses.asdict(cfg.model),
        "training": dataclasses.asdict(cfg.training),
        "data": dataclasses.asdict(cfg.data),
        "output_dir": cfg.output_dir,
        "device": cfg.device,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)

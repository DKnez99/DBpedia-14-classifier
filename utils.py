"""Shared runtime helpers for training, evaluation, and inference."""

import random
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import ProjectConfig


PATH_CONFIG_FIELDS = {
    "train_file",
    "test_file",
    "tokenizer_dir",
    "checkpoint_dir",
    "resume_from_checkpoint",
    "run_dir",
    "report_dir",
}


def set_seed(seed: int) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(config: ProjectConfig) -> None:
    """Configure PyTorch CPU threading for local training."""

    if config.torch_num_threads is not None:
        torch.set_num_threads(config.torch_num_threads)
    if config.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(config.torch_num_interop_threads)


def resolve_device(preferred_device: str | torch.device = "auto") -> torch.device:
    """Resolve an explicit or automatic PyTorch device selection."""

    if isinstance(preferred_device, torch.device):
        return preferred_device

    if preferred_device != "auto":
        device = torch.device(preferred_device)
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested, but PyTorch reports it is unavailable.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch reports it is unavailable.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_pin_memory(config: ProjectConfig) -> bool:
    """Return whether dataloaders should pin host memory."""

    return config.pin_memory and torch.cuda.is_available()


def config_to_checkpoint_dict(config: ProjectConfig) -> dict[str, Any]:
    """Convert a config dataclass to checkpoint-safe primitive values."""

    config_dict = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config_dict.items()
    }


def config_from_checkpoint(
    checkpoint: dict[str, Any],
    fallback: ProjectConfig,
) -> ProjectConfig:
    """Rebuild project config from checkpoint metadata."""

    checkpoint_config = checkpoint.get("config")
    if not checkpoint_config:
        return fallback

    valid_fields = {field.name for field in fields(ProjectConfig)}
    config_values = {}
    for key, value in checkpoint_config.items():
        if key not in valid_fields:
            continue
        if key in PATH_CONFIG_FIELDS and value is not None:
            config_values[key] = Path(value)
        elif key == "class_names":
            config_values[key] = tuple(value)
        else:
            config_values[key] = value

    return replace(fallback, **config_values)


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Remove bulky tensor values before writing metrics into checkpoints."""

    if not metrics:
        return {}
    return {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }

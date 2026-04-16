"""Config loading and small utilities shared across the pipeline."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

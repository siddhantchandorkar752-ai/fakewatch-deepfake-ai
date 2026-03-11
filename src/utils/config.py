import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_value(config: dict, *keys: str, default: Any = None) -> Any:
    try:
        result = config
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default

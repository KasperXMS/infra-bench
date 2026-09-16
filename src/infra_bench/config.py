from pathlib import Path
from typing import Any

import yaml

from .schemas import OperatorProfile


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value or {}


def load_operator_profiles(path: str | Path) -> dict[str, OperatorProfile]:
    raw = load_yaml(path)
    return {
        operator: OperatorProfile(operator=operator, **settings)
        for operator, settings in raw.items()
    }


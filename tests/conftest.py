from pathlib import Path

import pytest

from infra_bench.config import load_operator_profiles


@pytest.fixture
def profiles():
    return load_operator_profiles(Path("configs/operator_profiles.yaml"))


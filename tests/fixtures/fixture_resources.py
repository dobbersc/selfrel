from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    return Path(__file__).parents[1].resolve() / "resources"

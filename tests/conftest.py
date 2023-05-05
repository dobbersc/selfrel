from pathlib import Path

# noinspection PyUnresolvedReferences
import pytest


def path_to_plugin(path: Path) -> str:
    return str(path).replace("/", ".").replace("\\", ".").replace(".py", "")


# Load all fixtures from 'fixture_' modules in the 'tests/fixtures' package as pytest plugins.
# This exposes all fixtures defined in the detected modules to use in pytests.
# The setup is inspired by https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
# Documentation on the `pytest_plugins` variable:
# https://docs.pytest.org/en/latest/reference/reference.html#globalvar-pytest_plugins
pytest_plugins: list[str] = [path_to_plugin(fixture) for fixture in Path("tests/fixtures").glob("fixture_*.py")]

import traceback
from pathlib import Path
from typing import Any, Optional, Union

import yaml

try:
    import wandb
except ImportError as e:
    msg = "Please install selfrel with the wandb option 'selfrel[wandb]' before using any wandb sweeps."
    raise ImportError(msg) from e

__all__ = ["init", "agent"]


def _load_sweep_configuration(configuration_path: Union[str, Path]) -> dict[str, Any]:
    with Path(configuration_path).open(encoding="utf-8") as file:
        configuration: dict[str, Any] = yaml.safe_load(file)
    return configuration


def _submit_to_train() -> None:
    # Import `train` here to reduce import time for sweep initialization
    from selfrel.entry_points.train import train

    run = wandb.init()

    # noinspection PyBroadException
    try:
        train(**wandb.config, wandb_project=run.project)
    except Exception:
        # Print the traceback to stderr, so wandb logs the exception (https://github.com/wandb/wandb/issues/2387)
        traceback.print_exc()
        raise


def init(configuration: Union[str, Path], entity: Optional[str] = None, project: Optional[str] = None) -> str:
    """See `selfrel sweep init --help`."""
    sweep_configuration: dict[str, Any] = _load_sweep_configuration(configuration)
    sweep_id: str = wandb.sweep(sweep=sweep_configuration, entity=entity, project=project)

    wandb_project: str = sweep_configuration["project"] if project is None else project
    print("Create agents connected to this sweep with:")
    print(f"\tselfrel sweep agent {wandb_project} {sweep_id}")
    print("Create agents connected to this sweep on a specific cuda device with:")
    print(f"\tCUDA_VISIBLE_DEVICES=0 selfrel sweep agent {wandb_project} {sweep_id}")

    return sweep_id


def agent(project: str, sweep_id: str, entity: Optional[str] = None, count: Optional[int] = None) -> None:
    """See `selfrel sweep agent --help`"""
    wandb.agent(sweep_id, function=_submit_to_train, project=project, entity=entity, count=count)

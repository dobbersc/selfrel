import importlib
import inspect
import subprocess
from pathlib import Path
from typing import Any, Callable, Final

import pytest
from pytest_mock import MockerFixture

import selfrel
import selfrel.__main__

ENTRY_POINT_PARAMETERS: Final[tuple[dict[str, Any], ...]] = (
    {
        "entry_point": "selfrel.entry_points.export.cc_news.export_cc_news",
        "requires_ray": False,
        "required_arguments": "export cc-news",
        "custom_optional_arguments": (
            "--out-dir out-dir "
            "--no-export-metadata "
            "--dataset-slice :100 "
            "--max-sentence-length 100 "
            "--processes 4"
        ),
        "expected_call_arguments": {
            "out_dir": Path("out-dir"),
            "export_metadata": False,
            "dataset_slice": ":100",
            "max_sentence_length": 100,
            "processes": 4,
        },
    },
    {
        "entry_point": "selfrel.entry_points.export.knowledge_base.export_knowledge_base",
        "requires_ray": False,
        "required_arguments": "export knowledge-base --dataset dataset.conllup",
        "custom_optional_arguments": (
            "--out kb.db --entity-label-type ner --relation-label-type relation --no-create-relation-overview"
        ),
        "expected_call_arguments": {
            "dataset": Path("dataset.conllup"),
            "out": Path("kb.db"),
            "entity_label_type": "ner",
            "relation_label_type": "relation",
            "create_relation_overview": False,
        },
    },
    {
        "entry_point": "selfrel.entry_points.annotate.annotate",
        "requires_ray": True,
        "required_arguments": "annotate dataset.conllup",
        "custom_optional_arguments": (
            "--out dataset-annotated.conllup "
            "--model ner "
            "--label-type ner "
            "--abstraction-level span "
            "--batch-size 128 "
            "--num-actors 4 "
            "--num-cpus 16 "
            "--num-gpus 1 "
            "--buffer-size 4"
        ),
        "expected_call_arguments": {
            "dataset_path": Path("dataset.conllup"),
            "out": Path("dataset-annotated.conllup"),
            "model": "ner",
            "label_type": "ner",
            "abstraction_level": "span",
            "batch_size": 128,
            "num_actors": 4,
            "num_cpus": 16,
            "num_gpus": 1,
            "buffer_size": 4,
        },
    },
)

ENTRY_POINT_IDS: Final[tuple[str, ...]] = tuple(
    parameters["entry_point"].rsplit(".", 1)[1] for parameters in ENTRY_POINT_PARAMETERS
)

TEST_WITH_DEFAULT_ARGUMENTS: Final[tuple[str, ...]] = ("entry_point", "requires_ray", "required_arguments")
TEST_WITH_CUSTOM_ARGUMENTS: Final[tuple[str, ...]] = (
    "entry_point",
    "requires_ray",
    "required_arguments",
    "custom_optional_arguments",
    "expected_call_arguments",
)


class TestEntryPoints:
    @pytest.mark.parametrize(
        TEST_WITH_DEFAULT_ARGUMENTS,
        (
            [value for key, value in parameters.items() if key in TEST_WITH_DEFAULT_ARGUMENTS]
            for parameters in ENTRY_POINT_PARAMETERS
        ),
        ids=ENTRY_POINT_IDS,
    )
    def test_with_default_arguments(
        self, entry_point: str, requires_ray: bool, required_arguments: str, mocker: MockerFixture
    ) -> None:
        # Get argument defaults of the entry point's function
        module_name, function_name = entry_point.rsplit(".", 1)
        entry_point_fn: Callable[..., Any] = getattr(importlib.import_module(module_name), function_name)
        entry_point_signature: inspect.Signature = inspect.signature(entry_point_fn)
        default_arguments: dict[str, Any] = {
            key: value.default
            for key, value in entry_point_signature.parameters.items()
            if value.default is not inspect.Parameter.empty
        }

        # Mock
        entry_point_mock = mocker.patch(entry_point)
        ray_init_mock = mocker.patch("ray.init")

        # Run and assert
        selfrel.__main__.run(required_arguments.split())

        entry_point_mock.assert_called_once()
        assert entry_point_mock.call_args.args == ()
        assert {  # Only check default arguments
            key: value for key, value in entry_point_mock.call_args.kwargs.items() if key in default_arguments
        } == default_arguments

        if requires_ray:
            ray_init_mock.assert_called_once()

    @pytest.mark.parametrize(
        TEST_WITH_CUSTOM_ARGUMENTS,
        (
            [value for key, value in parameters.items() if key in TEST_WITH_CUSTOM_ARGUMENTS]
            for parameters in ENTRY_POINT_PARAMETERS
        ),
        ids=ENTRY_POINT_IDS,
    )
    def test_with_custom_arguments(
        self,
        entry_point: str,
        requires_ray: bool,
        required_arguments: str,
        custom_optional_arguments: str,
        expected_call_arguments: dict[str, object],
        mocker: MockerFixture,
    ) -> None:
        # Mock
        entry_point_mock = mocker.patch(entry_point)
        ray_init_mock = mocker.patch("ray.init")

        # Run and assert
        selfrel.__main__.run(required_arguments.split() + custom_optional_arguments.split())
        entry_point_mock.assert_called_once_with(**expected_call_arguments)

        if requires_ray:
            ray_init_mock.assert_called_once()


def test_run_as_module() -> None:
    """Tests if selfrel can be run as a Python module."""
    assert subprocess.call(["python", "-m", "selfrel", "--help"]) == 0  # noqa: S603, S607


def test_entry_point() -> None:
    """Tests if for selfrel the entry point script is installed (setup.cfg)."""
    assert subprocess.call(["selfrel", "--help"]) == 0  # noqa: S603, S607


def test_version() -> None:
    result: str = subprocess.check_output(["selfrel", "--version"]).decode("utf-8")  # noqa: S603, S607
    assert result == f"selfrel {selfrel.__version__}\n"

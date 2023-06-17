import importlib
import inspect
import platform
import subprocess
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Final

import pytest
from pytest_mock import MockerFixture

import selfrel
import selfrel.__main__
from selfrel.utils.iteration import iter_subclasses


class TestingEntryPointParameters(ABC):
    entry_point: str
    requires_ray: bool
    required_arguments: str
    custom_optional_arguments: str
    expected_call_arguments: dict[str, Any]


class ExportCCNews(TestingEntryPointParameters):
    entry_point: str = "selfrel.entry_points.export.cc_news.export_cc_news"
    requires_ray: bool = False
    required_arguments: str = "export cc-news"
    custom_optional_arguments: str = (
        "--out-dir out-dir "
        "--no-export-metadata "
        "--dataset-slice :100 "
        "--max-sentence-length 100 "
        "--processes 4"
    )
    expected_call_arguments: dict[str, Any] = {
        "out_dir": Path("out-dir"),
        "export_metadata": False,
        "dataset_slice": ":100",
        "max_sentence_length": 100,
        "processes": 4,
    }


class ExportKnowledgeBase(TestingEntryPointParameters):
    entry_point: str = "selfrel.entry_points.export.knowledge_base.export_knowledge_base"
    requires_ray: bool = False
    required_arguments: str = "export knowledge-base --dataset dataset.conllup"
    custom_optional_arguments: str = (
        "--out kb.db --entity-label-type ner --relation-label-type relation --no-create-relation-overview"
    )
    expected_call_arguments: dict[str, Any] = {
        "dataset": Path("dataset.conllup"),
        "out": Path("kb.db"),
        "entity_label_type": "ner",
        "relation_label_type": "relation",
        "create_relation_overview": False,
    }


class Annotate(TestingEntryPointParameters):
    entry_point: str = "selfrel.entry_points.annotate.annotate"
    requires_ray: bool = True
    required_arguments: str = "annotate dataset.conllup"
    custom_optional_arguments: str = (
        "--out dataset-annotated.conllup "
        "--model ner "
        "--label-type ner "
        "--abstraction-level span "
        "--batch-size 128 "
        "--num-actors 4 "
        "--num-cpus 16 "
        "--num-gpus 1 "
        "--buffer-size 4"
    )
    expected_call_arguments: dict[str, Any] = {
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
    }


class Train(TestingEntryPointParameters):
    entry_point: str = "selfrel.entry_points.train.train"
    requires_ray: bool = True
    required_arguments: str = "train conll04 --support-dataset support-dataset.conllup"
    custom_optional_arguments: str = (
        "--base-path base "
        "--down-sample-train 0.1 "
        "--transformer distilbert-base-uncased "
        "--max-epochs 3 "
        "--learning-rate 4e-5 "
        "--batch-size 16 "
        "--no-cross-augmentation "
        "--no-entity-pair-label-filter "
        "--encoding-strategy typed-entity-marker "
        "--self-training-iterations 2 "
        "--selection-strategy entropy "
        "--min-confidence 0.8 "
        "--min-occurrence 2 "
        "--max-occurrence 10 "
        "--distinct sentence "
        "--base 2 "
        "--max-entropy 0.2 "
        "--top-k 1000 "
        "--label-distribution no_relation=0.7 located_in=0.3 "
        "--precomputed-annotated-support-datasets dataset-1.conllup None "
        "--precomputed-relation-overviews relations-1.parquet None "
        "--num-actors 4 "
        "--num-cpus 16 "
        "--num-gpus 1 "
        "--buffer-size 8 "
        "--prediction-batch-size 64 "
        "--exclude-labels-from-evaluation no_relation "
        "--seed 8 "
    )
    expected_call_arguments: dict[str, Any] = {
        "corpus_name": "conll04",
        "support_dataset": Path("support-dataset.conllup"),
        "base_path": Path("base"),
        "down_sample_train": 0.1,
        "transformer": "distilbert-base-uncased",
        "max_epochs": 3,
        "learning_rate": 4e-5,
        "batch_size": 16,
        "cross_augmentation": False,
        "entity_pair_label_filter": False,
        "encoding_strategy": "typed-entity-marker",
        "self_training_iterations": 2,
        "selection_strategy": "entropy",
        "min_confidence": 0.8,
        "min_occurrence": 2,
        "max_occurrence": 10,
        "distinct": "sentence",
        "base": 2.0,
        "max_entropy": 0.2,
        "top_k": 1000,
        "label_distribution": {"no_relation": 0.7, "located_in": 0.3},
        "precomputed_annotated_support_datasets": [Path("dataset-1.conllup"), None],
        "precomputed_relation_overviews": [Path("relations-1.parquet"), None],
        "num_actors": 4,
        "num_cpus": 16.0,
        "num_gpus": 1.0,
        "buffer_size": 8,
        "prediction_batch_size": 64,
        "exclude_labels_from_evaluation": ["no_relation"],
        "seed": 8,
    }


ENTRY_POINT_IDS: Final[tuple[str, ...]] = tuple(
    entry_point_parameters.__name__ for entry_point_parameters in iter_subclasses(TestingEntryPointParameters)
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
            [getattr(entry_point_parameters, attribute) for attribute in TEST_WITH_DEFAULT_ARGUMENTS]
            for entry_point_parameters in iter_subclasses(TestingEntryPointParameters)
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
            [getattr(entry_point_parameters, attribute) for attribute in TEST_WITH_CUSTOM_ARGUMENTS]
            for entry_point_parameters in iter_subclasses(TestingEntryPointParameters)
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


def test_entry_point() -> None:
    """Tests if for selfrel the entry point script is installed (setup.cfg)."""
    assert subprocess.call(["selfrel", "--help"]) == 0  # noqa: S603, S607


def test_version() -> None:
    result: str = subprocess.check_output(["selfrel", "--version"]).decode("utf-8")  # noqa: S603, S607
    if platform.system() == "Windows":
        assert result == f"selfrel {selfrel.__version__}\r\n"
    else:
        assert result == f"selfrel {selfrel.__version__}\n"

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import importlib_resources

from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter

if TYPE_CHECKING:
    from importlib_resources.abc import Traversable


def call_export(args: argparse.Namespace) -> None:
    assert args.dataset == "cc-news"

    from selfrel.entry_points.export import export_cc_news

    export_cc_news(
        out_dir=args.out,
        export_metadata=args.export_metadata,
        dataset_slice=args.slice,
        max_sentence_length=args.max_sentence_length,
        processes=args.processes,
    )


def call_annotate(args: argparse.Namespace) -> None:
    import ray

    from selfrel.entry_points.annotate import annotate

    ray.init()
    annotate(
        dataset_path=args.dataset,
        out_path=args.out,
        model=args.model,
        label_type=args.label_type,
        abstraction_level=args.abstraction_level,
        batch_size=args.batch_size,
        num_actors=args.num_actors,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        buffer_size=args.buffer_size,
    )


def call_train(args: argparse.Namespace) -> None:
    import ray

    from selfrel.entry_points.train import train

    ray.init()
    train(
        args.corpus,
        args.support_dataset,
        down_sample_train=args.down_sample_train,
        base_path=args.base_path,
        transformer=args.transformer,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        cross_augmentation=args.cross_augmentation,
        entity_pair_label_filter=args.entity_pair_label_filter,
        encoding_strategy=args.encoding_strategy,
        self_training_iterations=args.self_training_iterations,
        selection_strategy=args.selection_strategy,
        num_actors=args.num_actors,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        buffer_size=args.buffer_size,
        prediction_batch_size=args.prediction_batch_size,
    )


def main() -> None:
    """The main entry-point."""
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(required=True)

    entrypoint_descriptions: Traversable = importlib_resources.files("selfrel.entry_points.descriptions")

    # Define "export" command arguments
    export = subparsers.add_parser(
        "export",
        help=(
            "Exports datasets segmented into paragraphs, sentences and tokens to the CoNLL-U Plus format. "
            "Currently, only CC-News is supported."
        ),
        description=(entrypoint_descriptions / "export.txt").read_text(encoding="utf-8"),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    export.set_defaults(func=call_export)
    export.add_argument(
        "dataset",
        choices=["cc-news"],
        help="The dataset to export. Currently, only CC-News is supported.",
    )
    export.add_argument(
        "-o",
        "--out",
        type=Path,
        default=".",
        help="The output directory for the processed articles.",
    )
    # noinspection PyTypeChecker
    export.add_argument(
        "--export-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, the article's metadata will be exported to 'OUT/metadata.json'.",
    )
    export.add_argument(
        "--slice",
        default=None,
        help=(
            "A huggingface datasets slice, e.g. ':100', '25%%:75%%'. "  # Double '%' to escape formatted string
            "Reference: https://huggingface.co/docs/datasets/v1.11.0/splits.html"
        ),
    )
    export.add_argument(
        "--max-sentence-length",
        type=int,
        default=None,
        help=(
            "Only export articles where all its sentences do not exceed the maximum sentence length. "
            "The original article IDs are preserved."
        ),
    )
    export.add_argument(
        "--processes",
        type=int,
        default=1,
        help="The number of processes for multiprocessing.",
    )

    # Define "annotate" command arguments
    annotate = subparsers.add_parser(
        "annotate",
        help="Annotates a CoNLL-U Plus dataset with token, span, relation or sentence labels.",
        description=(entrypoint_descriptions / "annotate.txt").read_text(encoding="utf-8"),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    annotate.set_defaults(func=call_annotate)
    annotate.add_argument("dataset", type=Path, help="The path to the CoNLL-U Plus dataset to annotate.")
    annotate.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "The output path of the annotated dataset. "
            "Per default, the dataset is exported to the same directory as the input dataset "
            "with the original name suffixed with the provided label-type, e.g. 'cc-news-ner.conllup'."
        ),
    )
    annotate.add_argument(
        "--model",
        default="flair/ner-english-large",
        help="A model path or identifier for a Flair classifier that annotates the provided dataset.",
    )
    annotate.add_argument(
        "--label-type",
        default=None,
        help="Overwrites the model's label-type. Per default, the model's default label-type is used.",
    )
    annotate.add_argument(
        "--abstraction-level",
        choices=["token", "span", "relation", "sentence"],
        default=None,
        help=(
            "Overwrites the model's abstraction level type. "
            "Per default, the abstraction level is inferred from the model."
        ),
    )
    annotate.add_argument("--batch-size", type=int, default=32, help="The model prediction batch size.")
    annotate.add_argument(
        "--num-actors",
        type=int,
        default=1,
        help="The number of Ray actors to start.",
    )
    annotate.add_argument(
        "--num-cpus",
        type=float,
        default=None,
        help="The number of CPUs required for each Ray actor.",
    )
    annotate.add_argument(
        "--num-gpus",
        type=float,
        default=1,
        help="The number of GPUs required for each Ray actor.",
    )
    annotate.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help=(
            "The buffer size of how many batches of sentences are loaded in memory at once."
            "Per default, the buffer size is NUM_ACTORS."
        ),
    )

    # Define "train" command arguments
    train = subparsers.add_parser(
        "train",
        help="TODO",
        description=(entrypoint_descriptions / "train.txt").read_text(encoding="utf-8"),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    train.set_defaults(func=call_train)
    train.add_argument("corpus", choices=["conll04"], help="TODO")
    train.add_argument("--support-dataset", type=Path, required=True, help="TODO")
    train.add_argument("--down-sample-train", type=float, default=None, help="TODO")
    train.add_argument("--base-path", type=Path, default=Path(), help="TODO")
    train.add_argument("--transformer", default="bert-base-uncased", help="TODO")
    train.add_argument("--max-epochs", type=int, default=10, help="TODO")
    train.add_argument("--learning-rate", type=float, default=5e-5, help="TODO")
    train.add_argument("--batch-size", type=int, default=32, help="TODO")
    # noinspection PyTypeChecker
    train.add_argument("--cross-augmentation", action=argparse.BooleanOptionalAction, default=True, help="TODO")
    # noinspection PyTypeChecker
    train.add_argument("--entity-pair-label-filter", action=argparse.BooleanOptionalAction, default=True, help="TODO")
    train.add_argument(
        "--encoding-strategy",
        choices=[
            "entity-mask",
            "typed-entity-mask",
            "entity-marker",
            "entity-marker-punct",
            "typed-entity-marker",
            "typed-entity-marker-punct",
        ],
        default="typed-entity-marker-punct",
        help="TODO",
    )
    train.add_argument(
        "--self-training-iterations",
        type=int,
        default=1,
        help="TODO",
    )
    train.add_argument(
        "--selection-strategy",
        choices=["prediction-confidence", "total-occurrence"],
        default="prediction-confidence",
        help="TODO",
    )
    train.add_argument("--confidence-threshold")  # TODO make these parameters as well (properly needs sentinel)
    train.add_argument("--occurrence-threshold")
    train.add_argument("--num-actors", type=int, default=1, help="TODO")
    train.add_argument("--num-cpus", type=float, default=None, help="TODO")
    train.add_argument("--num-gpus", type=float, default=1.0, help="TODO")
    train.add_argument("--buffer-size", type=int, default=None, help="TODO")
    train.add_argument("--prediction-batch-size", type=int, default=32, help="TODO")

    # Parse the args and call the dedicated function
    args: argparse.Namespace = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

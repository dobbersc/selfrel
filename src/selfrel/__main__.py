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

    # Parse the args and call the dedicated function
    args: argparse.Namespace = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

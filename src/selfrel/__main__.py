import argparse
from pathlib import Path

import importlib_resources
from importlib_resources.abc import Traversable

from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter


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


def main() -> None:
    """The main entry-point."""
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(required=True)

    entrypoint_descriptions: Traversable = importlib_resources.files("selfrel.resources.entrypoint_descriptions")

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
        help="If set, the article's metadata will be exported to '<OUT>/metadata.json'.",
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
        help="Only export articles where all its sentences do not exceed the maximum sentence length.",
    )
    export.add_argument(
        "--processes",
        type=int,
        default=1,
        help="The number of processes for multiprocessing.",
    )

    # Parse the args and call the dedicated function
    args: argparse.Namespace = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

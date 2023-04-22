import argparse
from pathlib import Path

import importlib_resources
from importlib_resources.abc import Traversable


def call_export(args: argparse.Namespace) -> None:
    assert args.dataset == "cc-news"

    from selfrel.cc_news import export_cc_news

    export_cc_news(
        out_dir=args.out,
        export_metadata=args.export_metadata,
        dataset_slice=args.slice,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size,
    )


def main() -> None:
    """The main entry-point."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    entrypoint_descriptions: Traversable = importlib_resources.files("selfrel.resources.entrypoint_descriptions")

    # Define "Export" command arguments
    export = subparsers.add_parser(
        "export",
        help=(
            "Exports datasets segmented into paragraphs, sentences and tokens to the CoNLL-U Plus format. "
            "Currently, only CC-News is supported."
        ),
        description=(entrypoint_descriptions / "export.txt").read_text(encoding="utf-8"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--num-processes",
        default=1,
        help="The number of processes for multiprocessing.",
    )
    export.add_argument(
        "--chunk-size",
        default=1,
        help="The chunk size for multiprocessing.",
    )

    # Parse the args and call the dedicated function
    args: argparse.Namespace = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

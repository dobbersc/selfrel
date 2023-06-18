import argparse
import sys
from pathlib import Path

import importlib_resources
from importlib_resources.abc import Traversable

import selfrel
from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter, StoreDictKeyPair, none_or_path

entrypoint_descriptions: Traversable = importlib_resources.files("selfrel.entry_points.descriptions")


def call_export_cc_news(args: argparse.Namespace) -> None:
    from selfrel.entry_points.export.cc_news import export_cc_news

    export_cc_news(
        out_dir=args.out_dir,
        export_metadata=args.export_metadata,
        dataset_slice=args.dataset_slice,
        max_sentence_length=args.max_sentence_length,
        processes=args.processes,
    )


def call_export_knowledge_base(args: argparse.Namespace) -> None:
    from selfrel.entry_points.export.knowledge_base import export_knowledge_base

    export_knowledge_base(
        dataset=args.dataset,
        out=args.out,
        entity_label_type=args.entity_label_type,
        relation_label_type=args.relation_label_type,
        create_relation_overview=args.create_relation_overview,
    )


def call_annotate(args: argparse.Namespace) -> None:
    import ray

    from selfrel.entry_points.annotate import annotate

    ray.init()
    annotate(
        dataset_path=args.dataset,
        out=args.out,
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
        corpus_name=args.corpus,
        support_dataset=args.support_dataset,
        base_path=args.base_path,
        down_sample_train=args.down_sample_train,
        train_with_dev=args.train_with_dev,
        transformer=args.transformer,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        cross_augmentation=args.cross_augmentation,
        entity_pair_label_filter=args.entity_pair_label_filter,
        encoding_strategy=args.encoding_strategy,
        self_training_iterations=args.self_training_iterations,
        selection_strategy=args.selection_strategy,
        min_confidence=args.min_confidence,
        min_occurrence=args.min_occurrence,
        max_occurrence=args.max_occurrence,
        distinct=args.distinct,
        base=args.base,
        max_entropy=args.max_entropy,
        top_k=args.top_k,
        label_distribution=args.label_distribution,
        precomputed_annotated_support_datasets=args.precomputed_annotated_support_datasets,
        precomputed_relation_overviews=args.precomputed_relation_overviews,
        num_actors=args.num_actors,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        buffer_size=args.buffer_size,
        prediction_batch_size=args.prediction_batch_size,
        evaluation_split=args.evaluation_split,
        use_final_model_for_evaluation=args.use_final_model_for_evaluation,
        exclude_labels_from_evaluation=args.exclude_labels_from_evaluation,
        seed=args.seed,
    )


def add_export_cc_news(export_subparsers) -> None:
    export_cc_news = export_subparsers.add_parser(
        "cc-news",
        help=(
            "Exports the CC-News dataset segmented into paragraphs, sentences and tokens to the CoNLL-U Plus format."
        ),
        description=(entrypoint_descriptions / "export" / "cc_news.txt").read_text(encoding="utf-8"),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    export_cc_news.set_defaults(func=call_export_cc_news)

    export_cc_news.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path(),
        help="The output directory for the processed articles.",
    )
    # noinspection PyTypeChecker
    export_cc_news.add_argument(
        "--export-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, the article's metadata will be exported to 'OUT/metadata.json'.",
    )
    export_cc_news.add_argument(
        "--dataset-slice",
        default=None,
        help=(
            "A huggingface datasets slice, e.g. ':100', '25%%:75%%'. "  # Double '%' to escape formatted string
            "Reference: https://huggingface.co/docs/datasets/v1.11.0/splits.html"
        ),
    )
    export_cc_news.add_argument(
        "--max-sentence-length",
        type=int,
        default=None,
        help=(
            "Only export articles where all its sentences do not exceed the maximum sentence length. "
            "The original article IDs are preserved."
        ),
    )
    export_cc_news.add_argument(
        "--processes",
        type=int,
        default=1,
        help="The number of processes for multiprocessing.",
    )


def add_export_knowledge_base(export_subparsers) -> None:
    export_knowledge_base = export_subparsers.add_parser(
        "knowledge-base",
        help="KB",
        description=(entrypoint_descriptions / "export" / "knowledge_base.txt").read_text(encoding="utf-8"),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    export_knowledge_base.set_defaults(func=call_export_knowledge_base)

    export_knowledge_base.add_argument("--dataset", type=Path, required=True, help="TODO")
    export_knowledge_base.add_argument("--out", type=Path, default=Path("knowledge-base.db"), help="TODO")
    export_knowledge_base.add_argument("--entity-label-type", default="ner", help="TODO")
    export_knowledge_base.add_argument("--relation-label-type", default="relation", help="TODO")
    export_knowledge_base.add_argument(
        "--create-relation-overview", action=argparse.BooleanOptionalAction, default=True, help="TODO"
    )


def add_annotate(subparsers) -> None:
    annotate = subparsers.add_parser(
        "annotate",
        help="Annotates a CoNLL-U Plus dataset with token, span, relation or sentence labels.",
        description=(entrypoint_descriptions / "annotate.txt").read_text(encoding="utf-8"),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    annotate.set_defaults(func=call_annotate)

    annotate.add_argument("dataset", type=Path, help="The path to the CoNLL-U Plus dataset to annotate.")
    annotate.add_argument(
        "-o",
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


def add_train(subparsers) -> None:
    train = subparsers.add_parser(
        "train",
        help="Trains a relation classification model using self-training.",
        description=(entrypoint_descriptions / "train.txt").read_text(encoding="utf-8"),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    train.set_defaults(func=call_train)

    train.add_argument(
        "corpus",
        choices=["conll04"],
        help="The identifier of the gold-annotated corpus. Currently, only CoNLL04 is supported.",
    )
    train.add_argument(
        "--support-dataset",
        type=Path,
        required=True,
        help="The path to the support dataset, i.e. the (large) unlabelled corpus.",
    )
    train.add_argument("--base-path", type=Path, default=Path(), help="The base directory for training artefacts.")
    train.add_argument(
        "--down-sample-train",
        type=float,
        default=None,
        help=(
            "This is a proportion to uniformly down-sample the training data on sentence-level. "
            "E.g. '0.1' down-samples the training data to 10%%."  # Double '%' to escape formatted string
        ),
    )
    train.add_argument(
        "--train-with-dev",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, the data from dev split will be added to the training data.",
    )
    train.add_argument(
        "--transformer",
        default="bert-base-uncased",
        help="The huggingface identifier of the underlying transformer architecture.",
    )
    train.add_argument("--max-epochs", type=int, default=10, help="The maximum number of epochs to train.")
    train.add_argument("--learning-rate", type=float, default=5e-5, help="The learning rate of the optimizer.")
    train.add_argument("--batch-size", type=int, default=32, help="Size of mini-batches during training.")
    # TODO: Further explain cross augmentation
    train.add_argument(
        "--cross-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, apply cross augmentation on the training data.",
    )
    train.add_argument(
        "--entity-pair-label-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set, all valid relation entity pair combinations are inferred from the training data "
            "and used as relation candidates. Otherwise, the relation candidate filter is disabled, "
            "i.e. the model classifies the relation for each entity pair in the cross-product of all entity pairs."
        ),
    )
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
        help=(
            "The identifier of an encoding strategy controlling the encoding of the head and tail entities "
            "in a sentence with a relation annotation. "
            "For more information, see the encoding strategies defined in 'flair.models.relation_classifier_model'."
        ),
    )
    train.add_argument(
        "--self-training-iterations",
        type=int,
        default=1,
        help="The number of self-training iterations.",
    )
    train.add_argument(
        "--selection-strategy",
        choices=["prediction-confidence", "occurrence", "entropy"],
        default="prediction-confidence",
        help=(
            "The identifier of a self-training selection strategy. "
            "For more information see the selection strategies defined in 'selfrel.selection_strategies'."
        ),
    )
    train.add_argument("--min-confidence", type=float, default=None, help="TODO")
    train.add_argument("--min-occurrence", type=int, default=None, help="TODO")
    train.add_argument("--max-occurrence", type=int, default=None, help="TODO")
    train.add_argument("--distinct", choices=["sentence", "in-between-text"], default=None, help="TODO")
    train.add_argument("--base", type=float, default=None, help="TODO")
    train.add_argument("--max-entropy", type=float, default=None, help="TODO")
    train.add_argument("--top-k", type=int, default=None, help="TODO")
    train.add_argument(
        "--label-distribution",
        action=StoreDictKeyPair,
        value_type=float,
        nargs="+",
        default=None,
        metavar="LABEL=WEIGHT",
        help="TODO",
    )
    train.add_argument(
        "--precomputed-annotated-support-datasets", nargs="*", default=(), type=none_or_path, help="TODO"
    )
    train.add_argument("--precomputed-relation-overviews", nargs="*", default=(), type=none_or_path, help="TODO")
    train.add_argument("--num-actors", type=int, default=1, help="TODO")
    train.add_argument("--num-cpus", type=float, default=None, help="TODO")
    train.add_argument("--num-gpus", type=float, default=1.0, help="TODO")
    train.add_argument("--buffer-size", type=int, default=None, help="TODO")
    train.add_argument("--prediction-batch-size", type=int, default=32, help="TODO")
    train.add_argument("--evaluation-split", choices=["train", "dev", "test"], default="test", help="TODO")
    train.add_argument(
        "--use-final-model-for-evaluation", action=argparse.BooleanOptionalAction, default=True, help="TODO"
    )
    train.add_argument("--exclude-labels-from-evaluation", nargs="*", default=None, help="TODO")
    train.add_argument("--seed", type=int, default=None, help="TODO")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="selfrel", formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {selfrel.__version__}")

    subparsers = parser.add_subparsers(required=True)

    export = subparsers.add_parser(
        "export",
        help="Exports artifacts, e.g. datasets or knowledge bases.",
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    export_subparsers = export.add_subparsers(required=True)

    add_export_cc_news(export_subparsers)
    add_export_knowledge_base(export_subparsers)

    add_annotate(subparsers)
    add_train(subparsers)

    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    """Parses the args and calls the dedicated function."""
    args: argparse.Namespace = parse_args(argv)
    args.func(args)


def main() -> None:
    """The main entry-point."""
    run(sys.argv[1:])


if __name__ == "__main__":
    main()

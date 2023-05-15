import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Final, Literal, Optional, Union

import ray
from flair.data import Corpus, Sentence
from flair.datasets import RE_ENGLISH_CONLL04, DataLoader
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import RelationClassifier

from selfrel.data import CoNLLUPlusDataset
from selfrel.trainer import SelfTrainer
from selfrel.utils.argparse import RawTextArgumentDefaultsHelpFormatter
from selfrel.utils.inspect_relations import infer_entity_pair_labels

logger: logging.Logger = logging.getLogger("flair")

_CORPORA: Final[dict[str, type[Corpus[Sentence]]]] = {
    "conll04": RE_ENGLISH_CONLL04,
}


def _load_corpus(corpus_name: str) -> Corpus[Sentence]:
    try:
        return _CORPORA[corpus_name]()
    except KeyError as e:
        msg = f"The corpus {repr(corpus_name)} is not supported"
        raise ValueError(msg) from e


def train(
    corpus_name: Literal["conll04"],
    support_dataset_path: Union[str, Path],
    base_path: Union[str, Path] = Path(),
    transformer: str = "bert-base-uncased",
    max_epochs: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 32,
    cross_augmentation: bool = True,
    entity_pair_label_filter: bool = True,
    encoding_strategy: Literal[
        "entity-mask",
        "typed-entity-mask",
        "entity-marker",
        "entity-marker-punct",
        "typed-entity-marker",
        "typed-entity-marker-punct",
    ] = "typed-entity-marker-punct",
    self_training_iterations: int = 1,
    selection_strategy: Literal["prediction-confidence"] = "prediction-confidence",
    num_actors: int = 1,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = 1,
    buffer_size: Optional[int] = None,
    prediction_batch_size: int = 32,
) -> None:
    hyperparameters: dict[str, Any] = locals()
    ray.init()

    # Step 1: Create the training data and support dataset
    # The relation extractor is *not* trained end-to-end.
    # A corpus for training the relation extractor requires annotated entities and relations.
    corpus: Corpus[Sentence] = _load_corpus(corpus_name)
    corpus.downsample(percentage=0.50, downsample_train=True)

    support_dataset: CoNLLUPlusDataset = CoNLLUPlusDataset(support_dataset_path, persist=False)

    # Step 2: Make the label dictionary from the corpus
    label_dictionary = corpus.make_label_dictionary("relation")

    # Step 3: Initialize fine-tunable transformer embedding
    embeddings = TransformerDocumentEmbeddings(model=transformer, layers="-1", fine_tune=True)

    # Step 4: Initialize relation classifier
    model: RelationClassifier = RelationClassifier(
        embeddings=embeddings,
        label_dictionary=label_dictionary,
        label_type="relation",
        entity_label_types="ner",
        entity_pair_labels=infer_entity_pair_labels(
            [batch[0] for batch in iter(DataLoader(corpus.train, batch_size=1))],
            relation_label_type="relation",
            entity_label_types="ner",
        )
        if entity_pair_label_filter
        else None,
        cross_augmentation=cross_augmentation,
        zero_tag_value="no_relation",
        encoding_strategy=getattr(
            sys.modules["flair.models.relation_classifier_model"], encoding_strategy.title().replace("-", "")
        )(),
        allow_unk_tag=False,
    )

    # Step 4: Initialize self-trainer
    trainer: SelfTrainer = SelfTrainer(
        model=model,
        annotated_corpus=corpus,
        unlabelled_dataset=CoNLLUPlusDataset("dev.conllup"),
        num_actors=num_actors,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        buffer_size=buffer_size,
        prediction_batch_size=prediction_batch_size,
    )
    logger.info("-" * 100)
    logger.info("Parameters:")
    for parameter, parameter_value in hyperparameters.items():
        logger.info(" - %s: %s", parameter, repr(parameter_value))
    logger.info("-" * 100)

    # Step 5: Run self-trainer
    trainer.train(
        base_path,
        self_training_iterations=self_training_iterations,
        selection_strategy=getattr(
            sys.modules["selfrel.selection_strategies"], selection_strategy.title().replace("-", "")
        )(),
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        mini_batch_size=batch_size,
        main_evaluation_metric=("macro avg", "f1-score"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument("corpus", choices=["conll04"], help="TODO")
    parser.add_argument("--support-dataset", type=Path, required=True, help="TODO")
    parser.add_argument("--base-path", type=Path, default=Path(), help="TODO")
    parser.add_argument("--transformer", default="bert-base-uncased", help="TODO")
    parser.add_argument("--max-epochs", type=int, default=10, help="TODO")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="TODO")
    parser.add_argument("--batch-size", type=int, default=32, help="TODO")
    # noinspection PyTypeChecker
    parser.add_argument("--cross-augmentation", action=argparse.BooleanOptionalAction, default=True, help="TODO")
    # noinspection PyTypeChecker
    parser.add_argument("--entity-pair-label-filter", action=argparse.BooleanOptionalAction, default=True, help="TODO")
    parser.add_argument(
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
    parser.add_argument("--self-training-iterations", type=int, default=1, help="TODO")
    parser.add_argument(
        "--selection-strategy", choices=["prediction-confidence"], default="prediction-confidence", help="TODO"
    )
    parser.add_argument("--num-actors", type=int, default=1, help="TODO")
    parser.add_argument("--num-cpus", type=float, default=None, help="TODO")
    parser.add_argument("--num-gpus", type=float, default=1.0, help="TODO")
    parser.add_argument("--buffer-size", type=int, default=None, help="TODO")
    parser.add_argument("--prediction-batch-size", type=int, default=32, help="TODO")

    args: argparse.Namespace = parser.parse_args()

    print(args)

    train(
        args.corpus,
        args.support_dataset,
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


if __name__ == "__main__":
    main()
